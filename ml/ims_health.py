"""Check the pump health score on bearings that were run until they failed.

    python3 ml/ims_health.py

Uses the IMS bearing dataset (University of Cincinnati, NASA Prognostics Data
Repository): four bearings on one shaft at 2000 rpm, a one-second 20 kHz
recording every 10 minutes until a bearing failed. Download "4. Bearings.zip"
from https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip, extract
IMS.7z and the three .rar files inside it, and put the folders here:

    ml/data/ims/1st_test/   (2156 files, 8 channels)
    ml/data/ims/2nd_test/   (984 files, 4 channels)
    ml/data/ims/3rd_test/   (6324 files, 4 channels; called 4th_test/txt in the archive)

Every recording goes through the same steps as the CWRU data in train.py
(184 Hz low-pass, 500 Hz, 256-sample windows, 10 log band energies), so the
health score is tested on what our MPU6050 node would see. Each bearing is
calibrated on its first 120 windows, exactly like a pump on the dashboard,
and then scored until the end of the test. Writes ml/ims_results.json.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import WINDOW, window_features  # noqa: E402
from health import POOR_BELOW, WATCH_BELOW, HealthModel, health_from_ratio, trend_eta  # noqa: E402
from train import to_edge_rate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMS_DIR = os.path.join(HERE, "data", "ims")
CACHE = os.path.join(HERE, "data", "ims_features.npz")
FS_IMS = 20000

# test folder -> (channel of each bearing, failed bearings and what failed)
TESTS = {
    "1st_test": ([0, 2, 4, 6], {3: "inner race", 4: "roller"}),
    "2nd_test": ([0, 1, 2, 3], {1: "outer race"}),
    "3rd_test": ([0, 1, 2, 3], {3: "outer race"}),
}


def snapshot_features(path):
    """Band energies of two 256-sample windows per recording, per bearing channel."""
    with open(path) as f:
        data = np.array(f.read().split(), dtype=float)
    columns = 8 if "1st_test" in path else 4
    data = data.reshape(-1, columns)
    out, rms = [], []
    for ch in range(columns):
        x = to_edge_rate(data[:, ch], FS_IMS)
        out.append([window_features(x[i:i + WINDOW]) for i in (0, WINDOW)])
        rms.append(float(np.sqrt(np.mean((data[:, ch] - data[:, ch].mean()) ** 2))))
    return out, rms


def stamp(name):
    return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S").timestamp()


def extract():
    if os.path.exists(CACHE):
        return dict(np.load(CACHE))
    arrays = {}
    for test in TESTS:
        folder = os.path.join(IMS_DIR, test)
        names = sorted(os.listdir(folder))
        print(f"{test}: {len(names)} recordings")
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(snapshot_features, [os.path.join(folder, n) for n in names], chunksize=16))
        arrays[test + "_features"] = np.array([r[0] for r in results], dtype=np.float32)
        arrays[test + "_rms"] = np.array([r[1] for r in results], dtype=np.float32)
        arrays[test + "_time"] = np.array([stamp(n) for n in names])
    np.savez_compressed(CACHE, **arrays)
    return arrays


# Reference: the same health score on the full 20 kHz signal (what a
# wide-band industrial sensor would see), to measure the cost of 184 Hz.
FULL_WINDOW = 2048
FULL_EDGES = [0, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
_full_freqs = np.fft.rfftfreq(FULL_WINDOW, 1 / FS_IMS)
_full_masks = [(_full_freqs >= a) & (_full_freqs < b) for a, b in zip(FULL_EDGES[:-1], FULL_EDGES[1:])]
_full_hann = np.hanning(FULL_WINDOW)


def full_band_features(path):
    with open(path) as f:
        data = np.array(f.read().split(), dtype=float)
    data = data.reshape(-1, 8 if "1st_test" in path else 4)
    out = []
    for ch in range(data.shape[1]):
        x = data[:, ch]
        rows = []
        for i in range(0, len(x) - FULL_WINDOW + 1, FULL_WINDOW * 5):     # 2 windows, like the MPU case
            s = x[i:i + FULL_WINDOW] - x[i:i + FULL_WINDOW].mean()
            p = np.abs(np.fft.rfft(s * _full_hann)) ** 2
            rows.append([np.log10(p[m].sum() + 1e-12) for m in _full_masks])
        out.append(rows)
    return out


def extract_full():
    cache = CACHE.replace(".npz", "_full.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    arrays = {}
    for test in TESTS:
        folder = os.path.join(IMS_DIR, test)
        names = sorted(os.listdir(folder))
        with ProcessPoolExecutor() as pool:
            arrays[test] = np.array(list(pool.map(full_band_features, [os.path.join(folder, n) for n in names],
                                                  chunksize=16)), dtype=np.float32)
    np.savez_compressed(cache, **arrays)
    return arrays


CAL_HOURS = 24        # calibration windows are spread over the first day of running
CAL_WINDOWS = 120     # same number as a dashboard calibration
SMOOTH = 6            # health points are the median of the last 6 recordings (1 hour)
SUSTAIN = 3           # a level counts once it holds for 3 points in a row
ETA_HOURS = 6         # straight-line fit over the last 6 hours for the time estimate
FALSE_ALARM_BEFORE_H = 48


def rolling_median(x, n=SMOOTH):
    return np.array([np.median(x[max(0, i - n + 1):i + 1]) for i in range(len(x))])


def first_sustained(flags, k=SUSTAIN):
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= k:
            return i
    return None


def health_curve(x, t):
    """Calibrate on windows spread over the first day, then score every recording."""
    n_day = int(np.searchsorted(t, t[0] + CAL_HOURS * 3600))
    idx = np.linspace(0, n_day - 1, CAL_WINDOWS // x.shape[1]).astype(int)
    model = HealthModel(x[idx].reshape(-1, x.shape[2]))
    raw = np.array([health_from_ratio(np.median(model.ratio(x[i]))) for i in range(len(x))])
    return rolling_median(raw), idx[-1] + 1


def rms_curve(rms, t, start_idx):
    """Classic baseline: RMS of the raw signal above mean + 3 sd of the same calibration recordings."""
    n_day = int(np.searchsorted(t, t[0] + CAL_HOURS * 3600))
    idx = np.linspace(0, n_day - 1, CAL_WINDOWS // 2).astype(int)
    base = rms[idx]
    return rolling_median(rms) > base.mean() + 3 * base.std()


def eta_errors(h, t, start):
    """Compare the straight-line estimate of time to POOR with what happened."""
    j0 = first_sustained(h[start:] < WATCH_BELOW)
    if j0 is None:
        return None
    j0 += start
    k = first_sustained(h[j0:] < POOR_BELOW)
    if k is None:
        return None
    j50 = j0 + k
    rel, over, given = [], 0, 0
    for j in range(j0, j50):
        window = (t >= t[j] - ETA_HOURS * 3600) & (t <= t[j])
        _, eta = trend_eta(t[window], h[window], POOR_BELOW)
        if eta is None:
            continue
        actual = (t[j50] - t[j]) / 3600
        given += 1
        over += eta / 3600 > actual
        rel.append(abs(eta / 3600 - actual) / max(actual, 0.5))
    return {"points": int(j50 - j0), "estimates": given, "overestimates": int(over),
            "median_relative_error": round(float(np.median(rel)), 2) if rel else None}


def main():
    mpu = extract()
    full = extract_full()
    results = {"dataset": "IMS bearing run-to-failure data (NASA Prognostics Data Repository)",
               "calibration": f"{CAL_WINDOWS} windows spread over the first {CAL_HOURS} h",
               "alarm": f"health below {WATCH_BELOW} (watch) or {POOR_BELOW} (poor) for {SUSTAIN} points, "
                        f"points = median of {SMOOTH} recordings",
               "bearings": [], "curves": {}}

    for test, (channels, failed) in TESTS.items():
        t = mpu[test + "_time"]
        end = t[-1]
        for bearing, ch in enumerate(channels, 1):
            row = {"test": test, "bearing": bearing, "failed": failed.get(bearing, "")}
            for name, x in (("mpu", mpu[test + "_features"][:, ch]), ("full", full[test][:, ch])):
                h, start = health_curve(x, t)
                for label, level in (("watch", WATCH_BELOW), ("poor", POOR_BELOW)):
                    i = first_sustained(h[start:] < level)
                    row[f"{name}_{label}_h_before_end"] = None if i is None else round((end - t[start + i]) / 3600, 1)
                if name == "mpu" and bearing in failed:
                    row["eta"] = eta_errors(h, t, start)
                    step = max(1, len(t) // 400)
                    results["curves"][f"{test} B{bearing}"] = {
                        "hours_before_end": [round((end - v) / 3600, 2) for v in t[::step]],
                        "health": [round(float(v), 1) for v in h[::step]],
                        "failed": failed[bearing],
                    }
            i = first_sustained(rms_curve(mpu[test + "_rms"][:, ch], t, 0)[start:])
            row["rms_alarm_h_before_end"] = None if i is None else round((end - t[start + i]) / 3600, 1)
            results["bearings"].append(row)
            print(row)

    # In the last two days the whole rig shakes because one bearing is failing,
    # so an alarm on a neighbouring bearing then is expected. Earlier ones are false.
    def summary(key):
        hits = [r[key] for r in results["bearings"] if r["failed"]]
        healthy = [r[key] for r in results["bearings"] if not r["failed"]]
        false = [h for h in healthy if h is not None and h > FALSE_ALARM_BEFORE_H]
        found = [h for h in hits if h is not None]
        return {"failed_detected": f"{len(found)}/{len(hits)}",
                "median_lead_h": round(float(np.median(found)), 1) if found else None,
                "healthy_false_alarms": f"{len(false)}/{len(healthy)}"}

    results["summary"] = {k: summary(k) for k in ("mpu_watch_h_before_end", "mpu_poor_h_before_end",
                                                  "full_watch_h_before_end", "rms_alarm_h_before_end")}
    with open(os.path.join(HERE, "ims_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
