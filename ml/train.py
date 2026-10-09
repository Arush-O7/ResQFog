"""Train the ResQFog anomaly model on the CWRU bearing dataset.

    python3 ml/train.py

Downloads the recordings it needs from the Case Western Reserve University
Bearing Data Center into ml/data/, converts each one to what our edge node
would see (MPU6050 184 Hz low-pass, 500 Hz sampling, 256-sample windows),
trains an Isolation Forest on normal windows only, and writes:

    ml/model.joblib    model, decision threshold and health baseline, loaded by fog_server.py
    ml/results.json    evaluation numbers used in the report
    ml/replay.npz      a few real windows, replayed by the --demo simulator

A Random Forest classifier is also trained, only as a supervised reference.
"""

import json
import os
import sys
import time
import urllib.request

import joblib
import numpy as np
import scipy.io as sio
from scipy.signal import butter, resample_poly, sosfilt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import DLPF_HZ, FEATURES, FS, WINDOW, window_features  # noqa: E402
from health import HealthModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
URL = "https://engineering.case.edu/sites/default/files/{}.mat"

# The normal baseline files are recorded at 48 kHz (they show the same
# machine peaks as the 48 kHz fault files); the drive-end fault files we
# use are recorded at 12 kHz.
NORMAL = {97: 0, 98: 1, 99: 2, 100: 3}          # file -> motor load (hp)
FAULTS = {
    # (fault, diameter in inches): files for loads 0, 1, 2, 3 hp
    ("inner", 0.007): [105, 106, 107, 108],
    ("ball", 0.007): [118, 119, 120, 121],
    ("outer", 0.007): [130, 131, 132, 133],
    ("inner", 0.014): [169, 170, 171, 172],
    ("ball", 0.014): [185, 186, 187, 188],
    ("outer", 0.014): [197, 198, 199, 200],
    ("inner", 0.021): [209, 210, 211, 212],
    ("ball", 0.021): [222, 223, 224, 225],
    ("outer", 0.021): [234, 235, 236, 237],
}

NORMAL_HOP = 32       # overlapping windows, the normal files are only ~5 s long
FAULT_HOP = 128
TRAIN_SHARE = 0.7     # first 70 % of each normal recording trains, the rest tests
FALSE_ALARM_TARGET = 0.01


def download(number):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{number}.mat")
    for attempt in range(4):
        if os.path.exists(path):
            return path
        try:
            print("downloading", number)
            urllib.request.urlretrieve(URL.format(number), path + ".part")
            os.replace(path + ".part", path)
        except OSError as e:
            print("  retry:", e)
            time.sleep(3)
    raise RuntimeError(f"could not download file {number}")


def load_signal(number):
    mat = sio.loadmat(download(number))
    key = next(k for k in mat if k.endswith("DE_time"))
    return mat[key].ravel()


def to_edge_rate(x, fs):
    """Low-pass like the MPU6050 DLPF, then resample to the edge sampling rate."""
    sos = butter(4, DLPF_HZ, fs=fs, output="sos")
    return resample_poly(sosfilt(sos, x), FS, fs)


def windows(x, hop):
    return [x[i:i + WINDOW] for i in range(0, len(x) - WINDOW + 1, hop)]


def build_dataset():
    normal_train, normal_test, faults = [], [], []

    for number, load in NORMAL.items():
        x = to_edge_rate(load_signal(number), 48000)
        cut = int(len(x) * TRAIN_SHARE)
        normal_train += [(w, load) for w in windows(x[:cut], NORMAL_HOP)]
        normal_test += [(w, load) for w in windows(x[cut:], NORMAL_HOP)]

    for (kind, size), files in FAULTS.items():
        for load, number in enumerate(files):
            x = to_edge_rate(load_signal(number), 12000)
            faults += [(w, kind, size, load) for w in windows(x, FAULT_HOP)]

    return normal_train, normal_test, faults


def rate(mask):
    return round(float(np.mean(mask)), 3)


def main():
    normal_train, normal_test, faults = build_dataset()
    print(f"windows: {len(normal_train)} normal train, {len(normal_test)} normal test, {len(faults)} fault")

    xt = np.array([window_features(w) for w, _ in normal_train])
    xn = np.array([window_features(w) for w, _ in normal_test])
    xf = np.array([window_features(w) for w, *_ in faults])

    # Isolation Forest, trained on normal data only
    model = IsolationForest(n_estimators=200, random_state=0).fit(xt)
    threshold = float(np.quantile(-model.score_samples(xt), 1 - FALSE_ALARM_TARGET))

    sn = -model.score_samples(xn)
    sf = -model.score_samples(xf)
    detected = sf > threshold
    false_alarm = sn > threshold

    tp, fp = int(detected.sum()), int(false_alarm.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = float(detected.mean())

    kinds = np.array([k for _, k, _, _ in faults])
    sizes = np.array([s for _, _, s, _ in faults])
    loads = np.array([l for *_, l in faults])

    iforest = {
        "trained_on": "normal windows only",
        "threshold": round(threshold, 4),
        "false_alarm_rate": rate(false_alarm),
        "detection_rate": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3),
        "roc_auc": round(float(roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sf))], np.r_[sn, sf])), 3),
        "by_fault": {f"{k} {s:.3f} in": rate(detected[(kinds == k) & (sizes == s)]) for k, s in FAULTS},
        "by_size": {f"{s:.3f} in": rate(detected[sizes == s]) for s in (0.007, 0.014, 0.021)},
        "by_type": {k: rate(detected[kinds == k]) for k in ("inner", "ball", "outer")},
        "by_load": {f"{l} hp": rate(detected[loads == l]) for l in range(4)},
    }

    # Random Forest reference: train on 0-2 hp, test on the unseen 3 hp load
    rows = [(f, "normal", l) for f, (_, l) in zip(np.r_[xt, xn], normal_train + normal_test)]
    rows += [(f, k, l) for f, (_, k, _, l) in zip(xf, faults)]
    train = [(f, k) for f, k, l in rows if l != 3]
    test = [(f, k) for f, k, l in rows if l == 3]
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit([f for f, _ in train], [k for _, k in train])
    pred = rf.predict([f for f, _ in test])
    truth = [k for _, k in test]
    classes = ["normal", "inner", "ball", "outer"]

    results = {
        "dataset": "CWRU Bearing Data Center, drive-end accelerometer, 0-3 hp",
        "preprocessing": f"{DLPF_HZ} Hz low-pass, resampled to {FS} Hz, {WINDOW}-sample windows",
        "features": FEATURES,
        "windows": {"normal_train": len(xt), "normal_test": len(xn), "fault": len(xf)},
        "isolation_forest": iforest,
        "random_forest_reference": {
            "split": "train on 0-2 hp, test on 3 hp",
            "accuracy": round(float(accuracy_score(truth, pred)), 3),
            "classes": classes,
            "confusion": confusion_matrix(truth, pred, labels=classes).tolist(),
        },
    }

    # health score baseline for the simulated pumps (calibrated pumps get their own)
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES, "source": "CWRU",
                 "health": HealthModel(xt)},
                os.path.join(HERE, "model.joblib"))

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Real windows for the demo simulator: normal (test part) and a mix of faults
    rng = np.random.default_rng(0)
    pick = lambda items, n: [items[i] for i in rng.choice(len(items), n, replace=False)]
    np.savez_compressed(
        os.path.join(HERE, "replay.npz"),
        normal=np.array([w for w, _ in pick(normal_test, 60)], dtype=np.float32),
        fault=np.array([w for w, *_ in pick(faults, 60)], dtype=np.float32),
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
