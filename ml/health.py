"""Pump health score, shared by the fog server and ml/ims_health.py.

The anomaly model answers "is this window abnormal?". The health score
answers "how far has this pump moved from its own normal running, and how
fast?", so it can be trended and extrapolated.

For each window we take the Mahalanobis distance of its 10 band energies
from the calibration windows (mean and a shrunk covariance), divided by the
95th percentile distance of those same windows. A ratio of 1 is the edge of
normal running. Faults grow roughly exponentially, so the score uses the
logarithm of the ratio:

    health = 100 * (1 - log(ratio) / log(RATIO_AT_ZERO)),  clipped to 0..100

100 means "like calibration"; 0 means the spectrum is RATIO_AT_ZERO times
further from normal than the normal spread.
"""

import math

import numpy as np
from sklearn.covariance import LedoitWolf

RATIO_AT_ZERO = 10.0     # distance ratio that maps to 0 % health
WATCH_BELOW = 75         # dashboard: below this the pump is "watch"
POOR_BELOW = 50          # below this "poor": plan maintenance now


class HealthModel:
    """Distance of band-energy windows from a pump's calibration windows."""

    def __init__(self, calibration):
        x = np.asarray(calibration, dtype=float)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-9
        self.precision = LedoitWolf().fit(x).precision_
        self.ref = float(np.quantile(self.distance(x), 0.95)) or 1.0

    def distance(self, x):
        d = np.atleast_2d(np.asarray(x, dtype=float)) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.precision, d))

    def ratio(self, x):
        """Distance divided by the 95th percentile of calibration (1 = edge of normal)."""
        return self.distance(x) / self.ref

    def band_z(self, x):
        """Per-band deviation from calibration in standard deviations (for the spectrum view)."""
        return (np.asarray(x, dtype=float) - self.mean) / self.std


def health_from_ratio(ratio):
    """Map a distance ratio (or the median of several) to 0..100."""
    r = max(float(ratio), 1.0)
    return max(0.0, min(100.0, 100.0 * (1.0 - math.log(r) / math.log(RATIO_AT_ZERO))))


def health_level(health):
    if health is None:
        return "UNKNOWN"
    if health < POOR_BELOW:
        return "POOR"
    if health < WATCH_BELOW:
        return "WATCH"
    return "GOOD"


def trend_eta(times, values, level, min_points=6):
    """Fit a straight line to recent health points and estimate when it reaches `level`.

    Returns (slope per second, seconds until the level is reached or None).
    Only a falling trend that explains most of the variation gives an estimate,
    so random scatter on a healthy pump does not produce a date.
    """
    t = np.asarray(times, dtype=float)
    h = np.asarray(values, dtype=float)
    if len(t) < min_points or t[-1] - t[0] <= 0:
        return None, None
    slope, intercept = np.polyfit(t - t[0], h, 1)
    fitted = intercept + slope * (t - t[0])
    total = float(np.sum((h - h.mean()) ** 2))
    r2 = 1.0 - float(np.sum((h - fitted) ** 2)) / total if total > 0 else 0.0
    now_value = fitted[-1]
    if slope >= 0 or r2 < 0.5:
        return float(slope), None
    if now_value <= level:
        return float(slope), 0.0
    return float(slope), float((level - now_value) / slope)
