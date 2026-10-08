"""Experiments reported in the ResQFog paper.

    python3 ml/experiments.py

Uses the same data preparation as ml/train.py and writes ml/experiments.json.
  E1  feature sets (ablation), Isolation Forest
  E2  anomaly detectors on band-energy features
  E3  Isolation Forest over 10 random seeds
  E4  leave-one-load-out
  E5  sensor bandwidth: full 12 kHz signal vs MPU6050-like 184 Hz / 500 Hz
  E6  timing on the fog computer
"""

import json
import os
import sys
import time

import joblib
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402
from features import FS, WINDOW, window_features  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_FA = 0.01


def shape_features(w):
    s = w - w.mean()
    rms = max(np.sqrt(np.mean(s ** 2)), 1e-9)
    mean_abs = max(np.mean(np.abs(s)), 1e-9)
    peak = np.max(np.abs(s))
    return [peak / rms, np.mean(s ** 4) / rms ** 4, np.mean(s ** 3) / rms ** 3, rms / mean_abs, peak / mean_abs]


def level_features(w):
    s = w - w.mean()
    return [np.log10(max(np.sqrt(np.mean(s ** 2)), 1e-9)), np.log10(max(np.max(np.abs(s)), 1e-9))]


def band_ratio_features(w):
    bands = 10 ** np.array(window_features(w))
    return list(bands / bands.sum())


FEATURE_SETS = {
    "shape (crest, kurtosis, skewness, shape, impulse)": shape_features,
    "level (log RMS, log peak)": level_features,
    "level + shape": lambda w: level_features(w) + shape_features(w),
    "relative band energy": band_ratio_features,
    "log band energy (proposed)": window_features,
}


def evaluate(scores_train, scores_normal, scores_fault):
    th = np.quantile(scores_train, 1 - TARGET_FA)
    det = scores_fault > th
    fa = scores_normal > th
    tp, fp = det.sum(), fa.sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = det.mean()
    auc = roc_auc_score(np.r_[np.zeros(len(scores_normal)), np.ones(len(scores_fault))], np.r_[scores_normal, scores_fault])
    return {
        "detection": round(float(rec), 4),
        "false_alarm": round(float(fa.mean()), 4),
        "precision": round(float(prec), 4),
        "f1": round(float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, 4),
        "auc": round(float(auc), 4),
    }


def iforest_scores(xt, xn, xf, seed=0):
    m = IsolationForest(n_estimators=200, random_state=seed).fit(xt)
    return -m.score_samples(xt), -m.score_samples(xn), -m.score_samples(xf)


def main():
    normal_train, normal_test, faults = T.build_dataset()
    wt = [w for w, _ in normal_train]
    wn = [w for w, _ in normal_test]
    wf = [w for w, *_ in faults]
    out = {"windows": {"normal_train": len(wt), "normal_test": len(wn), "fault": len(wf)}}

    # E1 feature ablation
    e1 = {}
    for name, fn in FEATURE_SETS.items():
        xt, xn, xf = (np.array([fn(w) for w in ws]) for ws in (wt, wn, wf))
        e1[name] = evaluate(*iforest_scores(xt, xn, xf))
        print("E1", name, e1[name])
    out["E1_features"] = e1

    xt, xn, xf = (np.array([window_features(w) for w in ws]) for ws in (wt, wn, wf))

    # E2 detectors
    detectors = {
        "Isolation Forest": lambda: IsolationForest(n_estimators=200, random_state=0),
        "One-Class SVM": lambda: make_pipeline(StandardScaler(), OneClassSVM(nu=0.01, gamma="scale")),
        "Local Outlier Factor": lambda: make_pipeline(StandardScaler(), LocalOutlierFactor(n_neighbors=20, novelty=True)),
    }
    e2 = {}
    for name, make in detectors.items():
        m = make().fit(xt)
        t0 = time.perf_counter()
        sn = -m.score_samples(xn)
        per_window_us = (time.perf_counter() - t0) / len(xn) * 1e6
        e2[name] = evaluate(-m.score_samples(xt), sn, -m.score_samples(xf)) | {"score_us_per_window": round(per_window_us, 1)}
        print("E2", name, e2[name])
    mu, sd = xt.mean(0), xt.std(0) + 1e-9
    z = lambda x: np.abs((x - mu) / sd).max(1)
    e2["Max z-score (baseline)"] = evaluate(z(xt), z(xn), z(xf))
    print("E2 z", e2["Max z-score (baseline)"])
    out["E2_detectors"] = e2

    # E3 seeds
    runs = [evaluate(*iforest_scores(xt, xn, xf, seed)) for seed in range(10)]
    out["E3_seeds"] = {k: {"mean": round(float(np.mean([r[k] for r in runs])), 4),
                           "std": round(float(np.std([r[k] for r in runs])), 4)} for k in runs[0]}
    print("E3", out["E3_seeds"])

    # E4 leave one load out
    e4 = {}
    for held in range(4):
        tr = np.array([window_features(w) for w, l in normal_train + normal_test if l != held])
        tn = np.array([window_features(w) for w, l in normal_test if l == held])
        tf = np.array([window_features(w) for w, _, _, l in faults if l == held])
        e4[f"{held} hp"] = evaluate(*iforest_scores(tr, tn, tf))
        print("E4", held, e4[f"{held} hp"])
    out["E4_leave_one_load_out"] = e4

    # E5 bandwidth: same 0.512 s windows at the original 12 kHz rate vs MPU6050-like
    full_len = int(WINDOW * 12000 / FS)        # 6144 samples = same duration
    hop_n, hop_f = full_len // 8, full_len // 2

    def full_rate(x, fs):
        return x if fs == 12000 else resample_poly(x, 12000, fs)

    def windows_full(x, hop):
        return [x[i:i + full_len] for i in range(0, len(x) - full_len + 1, hop)]

    ft, fn_, ff = [], [], []
    for number, load in T.NORMAL.items():
        x = full_rate(T.load_signal(number), 48000)
        cut = int(len(x) * T.TRAIN_SHARE)
        ft += windows_full(x[:cut], hop_n)
        fn_ += windows_full(x[cut:], hop_n)
    for (kind, size), files in T.FAULTS.items():
        for number in files:
            ff += windows_full(T.load_signal(number), hop_f)

    def full_band(w):
        s = w - w.mean()
        spec = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), 1 / 12000)
        edges = np.r_[0, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
        return [np.log10(spec[(fr >= a) & (fr < b)].sum() + 1e-12) for a, b in zip(edges[:-1], edges[1:])]

    e5 = {}
    for label, fn in (("shape features", shape_features), ("log band energy", full_band)):
        a, b, c = (np.array([fn(w) for w in ws]) for ws in (ft, fn_, ff))
        e5[f"12 kHz, {label}"] = evaluate(*iforest_scores(a, b, c))
    e5["500 Hz + 184 Hz LPF, shape features"] = e1["shape (crest, kurtosis, skewness, shape, impulse)"]
    e5["500 Hz + 184 Hz LPF, log band energy"] = e1["log band energy (proposed)"]
    out["E5_bandwidth"] = e5
    print("E5", e5)

    # E6 timing on this computer
    sample = wn[:100]
    t0 = time.perf_counter()
    feats = [window_features(w) for w in sample]
    feat_ms = (time.perf_counter() - t0) / len(sample) * 1000
    model = joblib.load(os.path.join(HERE, "model.joblib"))["model"]
    t0 = time.perf_counter()
    for f in feats:
        model.score_samples([f])
    score_ms = (time.perf_counter() - t0) / len(feats) * 1000
    t0 = time.perf_counter()
    IsolationForest(n_estimators=100, random_state=0).fit(xt[:120])
    calib_ms = (time.perf_counter() - t0) * 1000
    payload = len(",".join(str(int(round((1 + v) * 1000))) for v in sample[0])) + 120
    out["E6_timing"] = {
        "feature_ms_per_window": round(feat_ms, 3),
        "score_ms_per_window": round(score_ms, 3),
        "calibration_fit_ms_120_windows": round(calib_ms, 1),
        "payload_bytes_per_reading": payload,
    }
    print("E6", out["E6_timing"])

    with open(os.path.join(HERE, "experiments.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
