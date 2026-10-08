"""Check that the ESP32 band-energy code gives the same features as Python.

    python3 ml/check_band_features.py

Compiles edge/resqfog_edge/band_features.h with the computer's C compiler,
runs it on real CWRU windows from ml/replay.npz and compares the result with
ml/features.py. Also checks that rounding to the int16 values sent over LoRa
does not change the anomaly decisions.
"""

import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import FS, WINDOW, window_features  # noqa: E402

HARNESS = r"""
#include <stdio.h>
#include "band_features.h"
int main(void) {
    float x[%d], out[BAND_COUNT];
    while (1) {
        for (int i = 0; i < %d; i++) if (scanf("%%f", &x[i]) != 1) return 0;
        computeBandEnergies(x, %d, %d.0f, out);
        for (int b = 0; b < BAND_COUNT; b++) printf("%%.6f ", out[b]);
        printf("\n");
    }
}
""" % (WINDOW, WINDOW, WINDOW, FS)


def main():
    replay = np.load(os.path.join(HERE, "replay.npz"))
    windows = np.r_[replay["normal"], replay["fault"]] + 1.0     # magnitude in g, as on the ESP32

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "harness.c")
        exe = os.path.join(tmp, "harness")
        with open(src, "w") as f:
            f.write(HARNESS)
        subprocess.run(["cc", "-O2", "-I", os.path.join(ROOT, "edge", "resqfog_edge"), src, "-o", exe, "-lm"], check=True)
        data = "\n".join(" ".join(f"{v:.6f}" for v in w) for w in windows)
        out = subprocess.run([exe], input=data, capture_output=True, text=True, check=True).stdout

    c_feats = np.array([[float(v) for v in line.split()] for line in out.strip().splitlines()])
    py_feats = np.array([window_features(w) for w in windows])
    lora_feats = np.round(c_feats * 1000) / 1000          # int16 (x1000) in the LoRa packet

    model = joblib.load(os.path.join(HERE, "model.joblib"))
    th = model["threshold"]
    flag_py = -model["model"].score_samples(py_feats) > th
    flag_lora = -model["model"].score_samples(lora_feats) > th

    print(f"windows checked:            {len(windows)}")
    print(f"max |C - Python| feature:   {np.abs(c_feats - py_feats).max():.2e}")
    print(f"max |LoRa int16 - Python|:  {np.abs(lora_feats - py_feats).max():.2e}")
    print(f"same anomaly decision:      {int((flag_py == flag_lora).sum())}/{len(windows)}")


if __name__ == "__main__":
    main()
