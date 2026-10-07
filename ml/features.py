"""Vibration features shared by training (ml/train.py) and the fog server.

The ESP32 sends one window of acceleration magnitude with every reading.
The fog server turns it into these features with the same code that was
used for training, so the model always sees the same kind of input.
"""

import numpy as np

# Sampling on the edge node: MPU6050 with its 184 Hz low-pass filter,
# read every 2 ms, 256 readings per window (about half a second).
FS = 500
WINDOW = 256
DLPF_HZ = 184

# Energy in these frequency bands (Hz), on a log scale.
BAND_EDGES = [0, 20, 40, 60, 80, 100, 125, 150, 175, 200, 250]
FEATURES = [f"band_{a}_{b}" for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]

_freqs = np.fft.rfftfreq(WINDOW, 1 / FS)
_hann = np.hanning(WINDOW)
_masks = [(_freqs >= a) & (_freqs < b) for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]


def window_features(x):
    """Features of one window of acceleration (in g). Returns a list of floats."""
    s = np.asarray(x, dtype=float)
    s = s - s.mean()
    spectrum = np.abs(np.fft.rfft(s * _hann)) ** 2
    return [float(np.log10(spectrum[m].sum() + 1e-12)) for m in _masks]


def window_rms(x):
    s = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean((s - s.mean()) ** 2)))
