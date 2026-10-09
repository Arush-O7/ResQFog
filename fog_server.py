"""ResQFog fog server.

Receives telemetry from the ESP32 edge nodes installed on each pump, keeps
per-machine state for the dashboard, runs the ML anomaly model and the
health score on the vibration window that comes with each reading, records
the readings around every fault as CSV, and sends SMS alerts to the
maintenance team. Alerts must be acknowledged on the dashboard or they are
escalated. The pump registry, health history, alerts and maintenance log are
kept in SQLite (resqfog.db).

Run:
    python3 fog_server.py                  # real edge nodes, real alerts
    python3 fog_server.py --fleet          # real PUMP-01 + three simulated pumps
    python3 fog_server.py --demo           # four simulated pumps, no hardware, no alerts
    python3 fog_server.py --test-sms       # send one test SMS and exit
"""

import argparse
import base64
import csv
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory
from sklearn.ensemble import IsolationForest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "ml"))
from features import BAND_EDGES, WINDOW, window_features  # noqa: E402
from health import POOR_BELOW, WATCH_BELOW, HealthModel, health_from_ratio, health_level, trend_eta  # noqa: E402
from store import Store  # noqa: E402

load_dotenv()

# Alerts are sent as normal SMS, so the receiving phone needs no app.
#   android:       (default) an Android phone with a SIM running the free,
#                  open-source SMSGate app (sms-gate.app) in Local Server mode
#                  on the same Wi-Fi - sends the full alert text
#   gsm:           an A7670C / SIM800L GSM module on a USB serial port - full text
#   circuitdigest: free CircuitDigest Cloud SMS API (India, 100 SMS/month,
#                  fixed templates with two short fields, OTP-verified numbers)
SMS_BACKEND = os.getenv("SMS_BACKEND", "android").strip().lower()
CIRCUITDIGEST_API_KEY = os.getenv("CIRCUITDIGEST_API_KEY", "").strip()
CIRCUITDIGEST_URL = os.getenv("CIRCUITDIGEST_URL", "https://www.circuitdigest.cloud/api/v1/send_sms").strip()
# numbers to alert; falls back to MANAGER_PHONE from the earlier Twilio setup
SMS_TO = [n.strip() for n in (os.getenv("SMS_TO") or os.getenv("MANAGER_PHONE") or "").split(",") if n.strip()]
SMS_GATEWAY_URL = os.getenv("SMS_GATEWAY_URL", "").strip().rstrip("/")
SMS_GATEWAY_USER = os.getenv("SMS_GATEWAY_USER", "").strip()
SMS_GATEWAY_PASSWORD = os.getenv("SMS_GATEWAY_PASSWORD", "").strip()
GSM_PORT = os.getenv("GSM_PORT", "").strip()
GSM_BAUD = int(os.getenv("GSM_BAUD", "9600") or 9600)

# An alert nobody acknowledges on the dashboard within ESCALATE_AFTER seconds
# is sent again to these numbers (the supervisor); falls back to SMS_TO.
SMS_ESCALATE_TO = [n.strip() for n in os.getenv("SMS_ESCALATE_TO", "").split(",") if n.strip()]
ESCALATE_AFTER = int(os.getenv("ESCALATE_AFTER") or 600)

# Running hours between services (bearing greasing, inspection)
SERVICE_EVERY_HOURS = float(os.getenv("SERVICE_EVERY_HOURS") or 2000)

# Must match WARNING_THRESHOLD / CRITICAL_THRESHOLD in edge/resqfog_edge/resqfog_edge.ino
WARNING_THRESHOLD = 1.00
CRITICAL_THRESHOLD = 1.20

# Edge trips the motor after this many consecutive CRITICAL readings (same as TRIP_AFTER on the ESP32)
TRIP_AFTER = 3

# seconds between alerts of the same kind for the same pump; longer for the
# CircuitDigest free plan so 100 SMS a month last
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN") or (600 if SMS_BACKEND == "circuitdigest" else 60))
EDGE_TIMEOUT = 5           # seconds without telemetry before a node is shown offline

HISTORY_SIZE = 120
EVENT_SIZE = 50

FAULT_PRE_SAMPLES = 50
FAULT_POST_SAMPLES = 50
FAULT_LIST_SIZE = 20

DEFAULT_MACHINE = "PUMP-01"

DB_PATH = os.path.join(BASE_DIR, "resqfog.db")

# Health score: one point per pump every HEALTH_INTERVAL seconds (median of the
# windows in it), trended over TREND_WINDOW. --demo runs both six times faster.
HEALTH_INTERVAL = int(os.getenv("HEALTH_INTERVAL") or 60)
TREND_WINDOW = 6 * 3600
HEALTH_SMOOTH = 6             # health = median of the last 6 interval values (as in ml/ims_health.py)
HEALTH_SUSTAIN = 3            # points in a row before a level change counts

FAULT_DIR = os.path.join(BASE_DIR, "fault_logs")
DEMO_FAULT_DIR = os.path.join(FAULT_DIR, "demo")   # simulated pumps, kept apart from real data

STATES = ("NORMAL", "WARNING", "CRITICAL")

# ML anomaly detection (see ml/train.py)
ML_MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.joblib")
ML_PUMP_DIR = os.path.join(BASE_DIR, "ml", "pumps")     # per-pump models from calibration
CALIBRATION_WINDOWS = 120     # about 2 minutes of normal running
LONG_CALIBRATION_STEP = 30    # "long" calibration: one window every 30 s, so 1 hour
ML_VOTES = 5                  # look at the last 5 windows...
ML_NEEDED = 3                 # ...and call it an anomaly if 3 of them are abnormal
ML_MIN_SPEED = 10             # % - below this the motor is treated as stopped

# Pumping stations used by the simulator (--demo and --fleet), with
# approximate demo coordinates around Vellore
SIM_SITES = [
    ("PUMP-01", "VIT Main Sump", 12.96920, 79.15590),
    ("PUMP-02", "Katpadi Pump House", 12.97160, 79.13790),
    ("PUMP-03", "Gandhi Nagar OHT", 12.95470, 79.13550),
    ("PUMP-04", "Sathuvachari Borewell", 12.93800, 79.15750),
]


app = Flask(__name__)

if SMS_BACKEND == "circuitdigest":
    alerts_enabled = bool(SMS_TO and CIRCUITDIGEST_API_KEY)
elif SMS_BACKEND == "gsm":
    alerts_enabled = bool(SMS_TO and GSM_PORT)
else:
    alerts_enabled = bool(SMS_TO and SMS_GATEWAY_URL)

lock = threading.Lock()

server_start_time = time.time()

demo_mode = False

alert_state = {
    "status": "READY" if alerts_enabled else "DISABLED",
    "sent": 0,
    "lastSent": "--",
}

machines = {}

store = None      # Store, opened in main (an in-memory database for --demo)

try:
    base_model = joblib.load(ML_MODEL_PATH)
except (OSError, ValueError) as e:
    print("ML model not loaded (run python3 ml/train.py):", e)
    base_model = None


def now_str(epoch=None):
    return datetime.fromtimestamp(epoch or time.time()).strftime("%H:%M:%S")


def classify(vibration):
    if vibration >= CRITICAL_THRESHOLD:
        return "CRITICAL"
    if vibration >= WARNING_THRESHOLD:
        return "WARNING"
    return "NORMAL"


def format_duration(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def average(samples):
    values = [s["vibration"] for s in samples]
    return round(sum(values) / len(values), 3) if values else None


class Machine:
    """Everything the fog node knows about one pump / edge node."""

    def __init__(self, machine_id, site="", simulated=False, placeholder=False):
        self.id = machine_id
        self.site = site or machine_id
        self.simulated = simulated
        self.placeholder = placeholder

        self.vibration = 0.0
        self.pwm = 0
        self.speed_pct = 0
        self.status = "WAITING"
        self.motor = "RUNNING"
        self.timestamp = "--"
        self.last_seen = 0.0
        self.status_since = 0.0
        self.connected = False

        self.history = deque(maxlen=HISTORY_SIZE)
        self.arrivals = deque(maxlen=50)
        self.events = deque(maxlen=EVENT_SIZE)

        self.pre_buffer = deque(maxlen=FAULT_PRE_SAMPLES)
        self.capture = None
        self.captures = deque(maxlen=FAULT_LIST_SIZE)

        self.pending_command = None
        self.last_alerts = {}      # alert kind -> time it was last sent
        self.recovered = 0

        # registry (set on the dashboard); a GPS fix from the node, if fitted, overrides the location
        self.lat = None
        self.lon = None
        self.motor_kw = None
        self.installed = ""
        self.notes = ""
        self.registered_site = False
        self.gps_fix = False
        self.satellites = 0
        self.link = "Wi-Fi"
        self.rssi = None
        self.snr = None

        # ML: simulated pumps replay CWRU data, so they use the CWRU model.
        # A real pump gets its own model once it has been calibrated.
        self.ml_model = base_model if simulated else load_pump_model(machine_id)
        self.ml_calibration = None
        self.ml_flags = deque(maxlen=ML_VOTES)
        self.ml_score = None
        self.ml_state = "NORMAL" if self.ml_model else "NOT CALIBRATED"
        self.ml_cal_step = 1
        self.ml_cal_count = 0

        # health score: ratios of the windows in the current interval, then one point per interval
        self.ratios = []
        self.interval_vib = []
        self.interval_ml = []
        self.health = None
        self.raw_health = deque(maxlen=HEALTH_SMOOTH)
        self.health_points = deque()          # (time, health) inside TREND_WINDOW
        self.health_level = "UNKNOWN"
        self.level_run = 0
        self.last_features = None

        # running hours (motor on), total and at the last service
        self.run_seconds = 0.0
        self.service_seconds = 0.0
        self.serviced = False                 # tells the simulator a service was logged

        self.reset_stats()
        self.load_registry()

    def load_registry(self, hours=True):
        row = store.pump(self.id) if store else None
        if not row:
            return
        if row["site"]:
            self.site = row["site"]
        self.registered_site = bool(row["site"])
        if row["lat"] is not None and row["lon"] is not None:
            self.lat, self.lon = row["lat"], row["lon"]
        self.motor_kw = row["motor_kw"]
        self.installed = row["installed"] or ""
        self.notes = row["notes"] or ""
        if not hours:
            return
        self.run_seconds = row["run_seconds"] or 0.0
        self.service_seconds = row["service_seconds"] or 0.0
        since = time.time() - TREND_WINDOW
        for p in store.points(self.id, since):
            if p["health"] is not None:
                self.health_points.append((p["t"], p["health"]))
        if self.health_points:
            self.health = self.health_points[-1][1]
            self.health_level = health_level(self.health)

    @property
    def health_model(self):
        return self.ml_model.get("health") if self.ml_model else None

    def reset_stats(self):
        self.samples = 0
        self.total = 0.0
        self.peak = 0.0
        self.peak_time = "--"
        self.minimum = None
        self.alerts = 0
        self.trips = 0
        self.time_in_state = dict.fromkeys(STATES, 0.0)

    def add_event(self, level, title, detail="", epoch=None):
        self.events.appendleft({
            "time": now_str(epoch),
            "level": level,
            "title": title,
            "detail": detail,
        })

    @property
    def state(self):
        """State shown on the dashboard, which also covers offline and tripped."""
        if not self.last_seen:
            return "WAITING"
        if not self.connected:
            return "OFFLINE"
        if self.motor == "TRIPPED":
            return "TRIPPED"
        return self.status

    def update_site(self, data):
        if data.get("site") and not self.registered_site:
            self.site = str(data["site"])[:40]

        try:
            lat, lon = float(data["lat"]), float(data["lon"])
            fix = bool(int(data.get("gps", 0)))
            # coordinates typed into the sketch only count if the registry has none
            if (lat or lon) and (fix or self.lat is None):
                if fix and not self.gps_fix:
                    self.add_event("info", "GPS fix acquired", f"{lat:.5f}, {lon:.5f}")
                self.lat, self.lon, self.gps_fix = lat, lon, fix
                self.satellites = int(data.get("sats", 0) or 0)
        except (KeyError, TypeError, ValueError):
            pass

        if data.get("via") == "lora":
            if self.link != "LoRa":
                self.add_event("info", "Receiving over LoRa", "Readings arrive through the LoRa gateway")
            self.link = "LoRa"
            self.rssi = data.get("rssi")
            self.snr = data.get("snr")

    # ML anomaly detection on the vibration window sent with each reading

    def start_calibration(self, long=False):
        """Quick: 120 consecutive windows (2 min). Long: one window every 30 s for an hour,
        which covers more of the pump's normal variation (see ml/ims_health.py)."""
        self.ml_calibration = []
        self.ml_cal_step = LONG_CALIBRATION_STEP if long else 1
        self.ml_cal_count = 0
        self.ml_flags.clear()
        self.ml_state = "CALIBRATING"
        minutes = CALIBRATION_WINDOWS * self.ml_cal_step // 60
        self.add_event("info", "ML calibration started",
                       f"Collecting {CALIBRATION_WINDOWS} windows of normal running over about {minutes} min")

    def score_window(self, features):
        if self.motor == "TRIPPED" or self.speed_pct < ML_MIN_SPEED:
            if self.ml_state in ("NORMAL", "ANOMALY"):
                self.ml_state = "MOTOR STOPPED"
                self.ml_flags.clear()   # start fresh votes after a restart
            return
        if self.ml_state == "MOTOR STOPPED":
            self.ml_state = "NORMAL"


        if self.ml_calibration is not None:
            if self.ml_cal_count % self.ml_cal_step == 0:
                self.ml_calibration.append(features)
            self.ml_cal_count += 1
            if len(self.ml_calibration) >= CALIBRATION_WINDOWS:
                self.finish_calibration()
            return

        if not self.ml_model:
            return

        self.last_features = features
        if self.health_model is not None:
            self.ratios.append(float(self.health_model.ratio(features)[0]))

        score = float(-self.ml_model["model"].score_samples([features])[0])
        self.ml_score = score
        self.interval_ml.append(score)
        self.ml_flags.append(score > self.ml_model["threshold"])

        new_state = "ANOMALY" if sum(self.ml_flags) >= ML_NEEDED else "NORMAL"
        if new_state != self.ml_state:
            if new_state == "ANOMALY":
                detail = "Vibration pattern differs from normal running"
                if self.status == "NORMAL":
                    detail += ", before the threshold alarm"
                self.add_event("warning", "ML anomaly detected", detail)
                raise_alert(self, "ml")
            elif self.ml_state == "ANOMALY":
                self.add_event("normal", "ML pattern back to normal")
        self.ml_state = new_state

    def finish_calibration(self):
        data = np.array(self.ml_calibration)
        model = IsolationForest(n_estimators=100, random_state=0).fit(data)
        # 95th percentile per window; the 3-of-5 vote keeps false alarms near 0.1 %
        threshold = float(np.quantile(-model.score_samples(data), 0.95))
        self.ml_model = {"model": model, "threshold": threshold, "source": f"{self.id} calibration",
                         "health": HealthModel(data), "calibrated": time.time()}
        self.ml_calibration = None
        self.ml_state = "NORMAL"
        # the health trend restarts from the new baseline
        self.ratios.clear()
        self.raw_health.clear()
        self.health_points.clear()
        self.health, self.health_level, self.level_run = None, "UNKNOWN", 0
        try:
            os.makedirs(ML_PUMP_DIR, exist_ok=True)
            joblib.dump(self.ml_model, os.path.join(ML_PUMP_DIR, f"{self.id}.joblib"))
        except OSError as e:
            print("Could not save pump model:", e)
        self.add_event("info", "ML calibration finished", f"Model fitted on {len(data)} windows")

    def flush_interval(self, now):
        """Store one summary point for the last interval and check the health level."""
        if not self.interval_vib and not self.ratios:
            return
        health = None
        if self.ratios:
            self.raw_health.append(health_from_ratio(float(np.median(self.ratios))))
            health = float(np.median(self.raw_health))
        vib_avg = float(np.mean(self.interval_vib)) if self.interval_vib else None
        vib_max = float(np.max(self.interval_vib)) if self.interval_vib else None
        ml = float(np.mean(self.interval_ml)) if self.interval_ml else None
        self.ratios, self.interval_vib, self.interval_ml = [], [], []
        store.add_point(self.id, now, vib_avg, vib_max, health, ml, self.run_seconds)
        store.save_hours(self.id, self.run_seconds, self.service_seconds)
        if health is None:
            return

        self.health = health
        self.health_points.append((now, health))
        while self.health_points and self.health_points[0][0] < now - TREND_WINDOW:
            self.health_points.popleft()

        level = health_level(health)
        if level == self.health_level:
            self.level_run = 0
            return
        self.level_run += 1
        if self.level_run < HEALTH_SUSTAIN and self.health_level != "UNKNOWN":
            return
        previous, self.health_level, self.level_run = self.health_level, level, 0
        if previous == "UNKNOWN":
            return
        if level == "POOR":
            self.add_event("critical", "Pump health poor", f"Health {health:.0f} %, plan maintenance")
            raise_alert(self, "health")
        elif level == "WATCH" and previous == "GOOD":
            self.add_event("warning", "Pump health falling", f"Health {health:.0f} %, below {WATCH_BELOW} %")
        elif level == "GOOD" or (level == "WATCH" and previous == "POOR"):
            self.add_event("normal", "Pump health improved", f"Health {health:.0f} %")

    def health_summary(self, now):
        points = list(self.health_points)
        slope, eta = trend_eta([p[0] for p in points], [p[1] for p in points], POOR_BELOW)
        recent = store.points(self.id, now - TREND_WINDOW * 4, limit=400) if store else []
        return {
            "score": round(self.health, 1) if self.health is not None else None,
            "level": self.health_level,
            "perDay": round(slope * 86400, 1) if slope is not None else None,
            "etaPoor": round(eta) if eta is not None else None,
            "interval": HEALTH_INTERVAL,
            "trendWindow": TREND_WINDOW,
            "points": [{"t": now_str(p["t"]), "h": None if p["health"] is None else round(p["health"], 1),
                        "v": None if p["vib_avg"] is None else round(p["vib_avg"], 3)} for p in recent],
            "watch": WATCH_BELOW,
            "poor": POOR_BELOW,
        }

    def spectrum(self):
        """Band energies of the latest window against this pump's normal running."""
        labels = [f"{a}-{b} Hz" for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
        hm = self.health_model
        if self.last_features is None or hm is None:
            return {"labels": labels, "current": None}
        z = hm.band_z(self.last_features)
        return {
            "labels": labels,
            "current": [round(v, 3) for v in self.last_features],
            "baseline": [round(float(v), 3) for v in hm.mean],
            "spread": [round(float(v), 3) for v in hm.std],
            "z": [round(float(v), 1) for v in z],
            "flagged": [i for i, v in enumerate(z) if abs(v) > 3],
        }

    def hours(self):
        run_h = self.run_seconds / 3600
        since_h = (self.run_seconds - self.service_seconds) / 3600
        return {"run": round(run_h, 3), "sinceService": round(since_h, 3),
                "serviceEvery": SERVICE_EVERY_HOURS, "dueIn": round(SERVICE_EVERY_HOURS - since_h, 1)}

    def ml_summary(self):
        return {
            "state": self.ml_state,
            "score": round(self.ml_score, 3) if self.ml_score is not None else None,
            "threshold": round(self.ml_model["threshold"], 3) if self.ml_model else None,
            "model": ("CWRU bearing data" if self.ml_model.get("source") == "CWRU" else "calibrated on this pump")
            if self.ml_model else None,
            "progress": len(self.ml_calibration) if self.ml_calibration is not None else None,
            "needed": CALIBRATION_WINDOWS,
            "long": self.ml_cal_step > 1,
        }

    def check_timeout(self, now):
        if self.connected and not self.simulated and now - self.last_seen > EDGE_TIMEOUT:
            self.connected = False
            self.add_event("offline", "Edge node offline", f"No telemetry for {EDGE_TIMEOUT}s")
            if self.capture:
                self.finalize_capture(partial=True)

    def record_stats(self, vibration, epoch, status):
        self.samples += 1
        self.total += vibration
        if vibration > self.peak:
            self.peak = vibration
            self.peak_time = now_str(epoch)
        if self.minimum is None or vibration < self.minimum:
            self.minimum = vibration
        self.history.append({
            "e": epoch,
            "t": now_str(epoch),
            "v": round(vibration, 3),
            "s": status,
        })

    def ingest(self, vibration, pwm, status, motor=None, features=None):
        """Handle one live reading from the edge node."""
        now = time.time()

        status = str(status or "").upper()
        if status not in STATES:
            status = classify(vibration)

        previous = self.status

        if previous in STATES and self.last_seen:
            self.time_in_state[previous] += min(now - self.last_seen, 2.0)
            if self.motor == "RUNNING" and self.speed_pct >= ML_MIN_SPEED:
                self.run_seconds += min(now - self.last_seen, 2.0)

        if not self.connected:
            self.connected = True
            self.add_event("info", "Edge node connected", f"{self.id} is sending telemetry")

        self.speed_pct = max(0, min(100, int(pwm / 255 * 100)))
        self.vibration = vibration
        self.pwm = pwm
        self.status = status
        self.timestamp = now_str(now)
        self.last_seen = now
        self.arrivals.append(now)

        if motor in ("RUNNING", "TRIPPED") and motor != self.motor:
            if motor == "TRIPPED":
                self.trips += 1
                self.add_event(
                    "critical",
                    "Motor tripped by edge protection",
                    f"{TRIP_AFTER} consecutive critical readings, power cut at the pump",
                )
                raise_alert(self, "tripped")
            else:
                self.add_event("normal", "Motor restarted", "Trip cleared")
            self.motor = motor

        self.record_stats(vibration, now, status)
        self.interval_vib.append(vibration)

        is_new_fault = status == "CRITICAL" and previous != "CRITICAL"

        if status != previous:
            self.status_since = now
            detail = f"{vibration:.2f} g at {self.speed_pct}% speed"
            if status == "WARNING":
                self.add_event("warning", "Elevated vibration", detail)
            elif status == "CRITICAL":
                self.add_event("critical", "Critical vibration", detail)
            elif status == "NORMAL" and previous in STATES and self.motor != "TRIPPED":
                self.add_event("normal", "Returned to normal", f"{vibration:.2f} g")

        self.capture_sample({
            "epoch": now,
            "vibration": vibration,
            "motorSpeed": pwm,
            "motorSpeedPercent": self.speed_pct,
            "status": status,
        }, is_new_fault)

        if features is not None:
            self.score_window(features)

    def ingest_backlog(self, readings):
        """Readings the edge buffered while the fog server was unreachable."""
        now = time.time()
        added = []

        for r in readings:
            try:
                vibration = float(r["v"])
                epoch = now - float(r.get("age", 0)) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
            status = str(r.get("s", "")).upper()
            if status not in STATES:
                status = classify(vibration)
            self.record_stats(vibration, epoch, status)
            added.append((epoch, vibration, status))

        if not added:
            return 0

        # Buffered readings are older than the live ones already in the graph
        self.history = deque(sorted(self.history, key=lambda p: p["e"]), maxlen=HISTORY_SIZE)

        self.recovered += len(added)
        start, end = min(a[0] for a in added), max(a[0] for a in added)
        self.add_event(
            "info",
            "Buffered readings recovered",
            f"{len(added)} readings from {now_str(start)} to {now_str(end)}",
        )

        worst = max(added, key=lambda a: a[1])
        if worst[2] == "CRITICAL":
            self.add_event(
                "critical",
                "Critical reading during outage",
                f"{worst[1]:.2f} g at {now_str(worst[0])}",
            )

        return len(added)

    # Fault recording: 50 readings before and 50 after each new CRITICAL event

    def capture_sample(self, sample, is_new_fault):
        if self.capture:
            self.capture["post"].append(sample)
            self.capture["peak"] = max(self.capture["peak"], sample["vibration"])
            if len(self.capture["post"]) >= FAULT_POST_SAMPLES:
                self.finalize_capture()
        elif is_new_fault:
            self.capture = {
                "fault": sample,
                "pre": list(self.pre_buffer),
                "post": [],
                "peak": sample["vibration"],
            }
            self.add_event(
                "info",
                "Fault recording started",
                f"{len(self.pre_buffer)} samples before fault captured",
            )

        self.pre_buffer.append(sample)

    @property
    def fault_dir(self):
        return DEMO_FAULT_DIR if self.simulated else FAULT_DIR

    def finalize_capture(self, partial=False):
        capture, self.capture = self.capture, None
        fault = capture["fault"]
        folder = self.fault_dir

        stamp = datetime.fromtimestamp(fault["epoch"]).strftime("%Y%m%d_%H%M%S")
        filename = f"fault_{self.id}_{stamp}.csv"
        suffix = 2
        while os.path.exists(os.path.join(folder, filename)):
            filename = f"fault_{self.id}_{stamp}_{suffix}.csv"
            suffix += 1

        rows = (
            [("PRE", s) for s in capture["pre"]]
            + [("FAULT", fault)]
            + [("POST", s) for s in capture["post"]]
        )

        try:
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, filename), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "machine_id", "sample", "phase", "timestamp", "offset_s",
                    "vibration_g", "motor_pwm", "motor_speed_pct", "status",
                ])
                for index, (phase, s) in enumerate(rows, start=-len(capture["pre"])):
                    writer.writerow([
                        self.id,
                        index,
                        phase,
                        datetime.fromtimestamp(s["epoch"]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        f"{s['epoch'] - fault['epoch']:.3f}",
                        f"{s['vibration']:.4f}",
                        int(s["motorSpeed"]),
                        s["motorSpeedPercent"],
                        s["status"],
                    ])
        except OSError as e:
            self.add_event("critical", "Fault recording failed", str(e)[:120])
            print("Fault recording failed:", e)
            return

        self.captures.appendleft({
            "file": filename,
            "time": datetime.fromtimestamp(fault["epoch"]).strftime("%Y-%m-%d %H:%M:%S"),
            "faultValue": round(fault["vibration"], 3),
            "peak": round(capture["peak"], 3),
            "preAvg": average(capture["pre"]),
            "postAvg": average(capture["post"]),
            "pre": len(capture["pre"]),
            "post": len(capture["post"]),
            "partial": partial,
        })

        self.add_event(
            "info",
            "Fault recording saved",
            filename + (" (partial, edge went offline)" if partial else ""),
        )
        print("Fault recording saved:", os.path.join(folder, filename))

    def summary(self, now):
        return {
            "id": self.id,
            "site": self.site,
            "state": self.state,
            "status": self.status,
            "motor": self.motor,
            "vibration": round(self.vibration, 3),
            "speed": self.speed_pct,
            "online": self.connected,
            "simulated": self.simulated,
            "alerts": self.alerts,
            "lastSeenAgo": round(now - self.last_seen, 1) if self.last_seen else None,
            "ml": self.ml_state,
            "link": self.link,
            "health": round(self.health) if self.health is not None else None,
            "healthLevel": self.health_level,
            "lat": self.lat,
            "lon": self.lon,
        }

    def detail(self, now):
        recent = [t for t in self.arrivals if now - t <= 10]
        total_state_time = sum(self.time_in_state.values())
        last = self.last_alerts.get("critical", 0)
        cooldown = max(0, int(ALERT_COOLDOWN - (now - last))) if last else 0

        return {
            "machine": self.summary(now) | {
                "trips": self.trips,
                "lat": self.lat,
                "lon": self.lon,
                "gpsFix": self.gps_fix,
                "satellites": self.satellites,
                "mapsLink": maps_link(self.lat, self.lon) if self.lat is not None else None,
                "rssi": self.rssi,
                "snr": self.snr,
                "recovered": self.recovered,
                "alertCooldown": cooldown,
            },
            "current": {
                "vibration": self.vibration,
                "motorSpeed": self.pwm,
                "motorSpeedPercent": self.speed_pct,
                "status": self.status,
                "timestamp": self.timestamp,
                "statusFor": format_duration(now - self.status_since) if self.status_since else "--",
            },
            "edge": {
                "online": self.connected,
                "lastSeenAgo": round(now - self.last_seen, 1) if self.last_seen else None,
                "rate": round(len(recent) / 10, 1),
            },
            "stats": {
                "samples": self.samples,
                "avg": self.total / self.samples if self.samples else 0,
                "max": self.peak,
                "maxTime": self.peak_time,
                "min": self.minimum or 0,
                "alerts": self.alerts,
            },
            "statusTime": {
                s: round(100 * self.time_in_state[s] / total_state_time, 1) if total_state_time else 0
                for s in STATES
            },
            "history": [{"t": p["t"], "v": p["v"]} for p in self.history],
            "events": list(self.events)[:20],
            "ml": self.ml_summary(),
            "health": self.health_summary(now),
            "spectrum": self.spectrum(),
            "hours": self.hours(),
            "registry": {"site": self.site, "lat": self.lat, "lon": self.lon, "motorKw": self.motor_kw,
                         "installed": self.installed, "notes": self.notes},
            "openAlerts": [alert_row(a) for a in store.alerts(self.id, open_only=True)],
            "alertHistory": [alert_row(a) for a in store.alerts(self.id, limit=15)],
            "maintenance": [maintenance_row(r) for r in store.maintenance(self.id, limit=15)],
            "faults": {
                "preSamples": FAULT_PRE_SAMPLES,
                "postSamples": FAULT_POST_SAMPLES,
                "recording": {
                    "time": now_str(self.capture["fault"]["epoch"]),
                    "pre": len(self.capture["pre"]),
                    "post": len(self.capture["post"]),
                } if self.capture else None,
                "saved": list(self.captures),
            },
        }


def load_pump_model(machine_id):
    path = os.path.join(ML_PUMP_DIR, f"{machine_id}.joblib")
    try:
        return joblib.load(path) if os.path.exists(path) else None
    except (OSError, ValueError):
        return None


def maps_link(lat, lon):
    return f"https://maps.google.com/?q={lat:.5f},{lon:.5f}"


def date_str(epoch):
    return datetime.fromtimestamp(epoch).strftime("%d %b %H:%M") if epoch else None


def alert_row(a):
    return {
        "id": a["id"], "pump": a["pump"], "kind": a["kind"], "title": a["title"], "detail": a["detail"],
        "time": date_str(a["t"]), "last": date_str(a["last_t"]), "count": a["count"], "sms": a["sms"] or "--",
        "escalated": date_str(a["escalated"]), "ackBy": a["ack_by"], "ackTime": date_str(a["ack_t"]),
        "ackNote": a["ack_note"] or "", "openFor": round(time.time() - a["t"]) if not a["ack_t"] else None,
    }


def maintenance_row(r):
    return {"id": r["id"], "pump": r["pump"], "time": date_str(r["t"]), "technician": r["technician"],
            "action": r["action"], "notes": r["notes"] or "", "runHours": round(r["run_hours"] or 0, 1),
            "service": bool(r["service"])}


def parse_features(data):
    """Band-energy features for one reading.

    A Wi-Fi node sends its raw window ("w", milli-g) and the fog computes the
    features; a LoRa node computes them itself and the gateway sends them as "f".
    """
    f = data.get("f")
    if isinstance(f, list) and len(f) == 10:
        try:
            return [float(v) for v in f]
        except (TypeError, ValueError):
            return None
    window = parse_window(data)
    return window_features(window) if window is not None else None


def parse_window(data):
    """The edge sends its window as integers in milli-g."""
    raw = data.get("w")
    if not isinstance(raw, list) or len(raw) != WINDOW:
        return None
    try:
        return [float(v) / 1000.0 for v in raw]
    except (TypeError, ValueError):
        return None


def get_machine(machine_id):
    machine_id = str(machine_id or DEFAULT_MACHINE).strip().upper()[:16] or DEFAULT_MACHINE
    m = machines.get(machine_id)
    if m is None:
        m = machines[machine_id] = Machine(machine_id)
    m.placeholder = False
    return m


def visible_machines():
    shown = [m for m in machines.values() if not m.placeholder] or list(machines.values())
    return sorted(shown, key=lambda m: m.id)


def load_saved_captures():
    """Show fault CSVs from earlier runs on the dashboard after a restart."""
    for folder in (FAULT_DIR, DEMO_FAULT_DIR):
        if os.path.isdir(folder):
            load_captures_from(folder)


def load_captures_from(folder):
    files = sorted(
        (f for f in os.listdir(folder) if f.startswith("fault_") and f.endswith(".csv")),
        key=lambda f: re.findall(r"\d{8}_\d{6}", f)[-1:],
    )

    for filename in files:
        # fault_PUMP-01_20261007_075007.csv, or fault_20261007_075007.csv from before machine IDs
        match = re.match(r"fault_(.+?)_\d{8}_\d{6}(_\d+)?\.csv$", filename)
        machine_id = match.group(1) if match else DEFAULT_MACHINE

        try:
            with open(os.path.join(folder, filename), newline="") as f:
                rows = list(csv.DictReader(f))
            fault = next(r for r in rows if r["phase"] == "FAULT")
            pre = [{"vibration": float(r["vibration_g"])} for r in rows if r["phase"] == "PRE"]
            post = [{"vibration": float(r["vibration_g"])} for r in rows if r["phase"] == "POST"]
        except (OSError, KeyError, ValueError, StopIteration):
            continue

        m = machines.get(machine_id)
        if m is None or m.fault_dir != folder:
            continue

        m.captures.appendleft({
            "file": filename,
            "time": fault["timestamp"][:19],
            "faultValue": float(fault["vibration_g"]),
            "peak": max(float(r["vibration_g"]) for r in rows),
            "preAvg": average(pre),
            "postAvg": average(post),
            "pre": len(pre),
            "post": len(post),
            "partial": len(post) < FAULT_POST_SAMPLES,
        })


# ---------------------------------------------------------------- alerts (SMS)

ALERT_TITLES = {
    "critical": "CRITICAL vibration",
    "tripped": "MOTOR TRIPPED",
    "ml": "EARLY WARNING (ML)",
    "health": "HEALTH POOR",
}

ALERT_ACTIONS = {
    "critical": "Inspect the pump. Motor trips if this lasts 3 readings.",
    "tripped": "Inspect, then restart from the dashboard or speed knob.",
    "ml": "Level still normal, pattern abnormal. Plan an inspection.",
    "health": "Health keeps falling. Plan maintenance soon.",
}


def sms_safe(text):
    """Keep to plain GSM characters so every phone shows the message correctly."""
    return text.encode("ascii", "ignore").decode()


def alert_text(m, kind, vibration=None):
    """The SMS the maintenance team receives."""
    vibration = m.vibration if vibration is None else vibration
    lines = [
        f"ResQFog ALERT: {ALERT_TITLES[kind]}",
        f"Pump: {m.id} ({m.site})",
        f"Vibration: {vibration:.2f} g (critical {CRITICAL_THRESHOLD:.2f} g)",
        f"Motor: {'TRIPPED' if m.motor == 'TRIPPED' else f'running {m.speed_pct}%'}",
    ]
    if m.ml_score is not None and m.ml_model:
        lines.append(f"ML: {m.ml_state.lower()}, score {m.ml_score:.2f}/{m.ml_model['threshold']:.2f}")
    if m.health is not None:
        lines.append(f"Health: {m.health:.0f}% ({m.health_level.lower()})")
    lines.append(f"Time: {datetime.now().strftime('%d-%b %H:%M:%S')}")
    if m.lat is not None:
        lines.append(f"Location: {m.lat:.5f},{m.lon:.5f} ({'GPS' if m.gps_fix else 'installed'})")
        lines.append(f"Map: {maps_link(m.lat, m.lon)}")
    if m.link == "LoRa" and m.rssi is not None:
        lines.append(f"Link: LoRa {m.rssi} dBm")
    lines.append(ALERT_ACTIONS[kind])
    return sms_safe("\n".join(lines))


def split_sms(text, size=153):
    """Split into SMS-sized parts at line breaks (for the GSM module)."""
    parts, current = [], ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= size - 6:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = line[: size - 6]
    if current:
        parts.append(current)
    if len(parts) > 1:
        parts = [f"({i}/{len(parts)}) {p}" for i, p in enumerate(parts, 1)]
    return parts


def send_via_android(numbers, text):
    """SMS Gateway for Android, local server mode (Basic auth, JSON)."""
    body = json.dumps({"textMessage": {"text": text}, "phoneNumbers": numbers}).encode()
    auth = base64.b64encode(f"{SMS_GATEWAY_USER}:{SMS_GATEWAY_PASSWORD}".encode()).decode()
    last_error = None
    # the endpoint is /message in older app versions and /messages in newer ones
    for path in ("/message", "/messages"):
        request = urllib.request.Request(
            SMS_GATEWAY_URL + path, data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if 200 <= response.status < 300:
                    return
        except urllib.error.HTTPError as e:
            last_error = f"gateway replied {e.code}"
            if e.code == 404:
                continue
            raise RuntimeError(last_error)
        except urllib.error.URLError as e:
            raise RuntimeError(f"phone gateway not reachable ({e.reason})")
    raise RuntimeError(last_error or "gateway did not accept the message")


class GsmModem:
    """Sends SMS with AT commands through a SIM800L / A7670 module."""

    def __init__(self, stream):
        self.stream = stream

    def command(self, text, expect=("OK",), timeout=5.0):
        self.stream.write(text.encode())
        return self.wait(expect, timeout)

    def wait(self, expect, timeout):
        reply, end = "", time.time() + timeout
        while time.time() < end:
            chunk = self.stream.read(64)
            if chunk:
                reply += chunk.decode(errors="ignore")
                if "ERROR" in reply:
                    raise RuntimeError("modem: " + reply.strip().splitlines()[-1])
                if any(e in reply for e in expect):
                    return reply
            else:
                time.sleep(0.05)
        raise RuntimeError("modem did not answer (check wiring, power and SIM)")

    def send(self, number, text):
        self.command("AT\r")
        self.command("AT+CMGF=1\r")                       # text mode
        self.command(f'AT+CMGS="{number}"\r', expect=(">",))
        self.command(text + "\x1a", expect=("+CMGS",), timeout=30.0)


gsm_lock = threading.Lock()


def send_via_gsm(numbers, text):
    import serial   # pyserial, only needed for the GSM module
    with gsm_lock, serial.Serial(GSM_PORT, GSM_BAUD, timeout=0.2) as port:
        modem = GsmModem(port)
        for number in numbers:
            for part in split_sms(text):
                modem.send(number, part)


# CircuitDigest templates (two fields each, at most 30 letters/digits):
#   113 "The {#var#} requires maintenance. Detected issue: {#var#}."
#   110 "The device {#var#} is currently located at {#var#}."
#   101 "Your {#var#} is currently at {#var#}."
CD_ISSUES = {
    "critical": lambda m, v: f"critical vibration {v:.2f}g",
    "tripped": lambda m, v: f"motor tripped at {v:.2f}g",
    "ml": lambda m, v: f"ML early warning {m.ml_score:.2f}" if m.ml_score is not None else "ML early warning",
    "health": lambda m, v: f"health {m.health:.0f} percent" if m.health is not None else "health poor",
}


def cd_field(text):
    """Letters, digits, spaces and dots only, 30 characters at most."""
    cleaned = re.sub(r"[^A-Za-z0-9 .]", "", text.replace("-", ""))
    return re.sub(r" +", " ", cleaned).strip()[:30]


def cd_messages(m, kind, vibration=None):
    """The template SMS for one alert: the issue, then the location if known."""
    vibration = m.vibration if vibration is None else vibration
    pump = cd_field(f"pump {m.id} {m.site}")
    messages = [(113, pump, cd_field(CD_ISSUES[kind](m, vibration)))]
    if m.lat is not None and kind not in ("ml", "health"):
        messages.append((110, cd_field(f"pump {m.id}"), cd_field(f"{m.lat:.5f} {m.lon:.5f}")))
    return messages


def cd_number(number):
    digits = re.sub(r"\D", "", number)
    return "91" + digits if len(digits) == 10 else digits


def send_via_circuitdigest(numbers, messages):
    for template, var1, var2 in messages:
        for number in numbers:
            body = json.dumps({"mobiles": cd_number(number), "var1": var1, "var2": var2}).encode()
            # the docs write the template parameter as "id", working examples use "ID"
            request = urllib.request.Request(
                f"{CIRCUITDIGEST_URL}?ID={template}&id={template}", data=body,
                headers={"Content-Type": "application/json", "Authorization": CIRCUITDIGEST_API_KEY},
            )
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    if response.status != 200:
                        raise RuntimeError(f"CircuitDigest replied {response.status}")
            except urllib.error.HTTPError as e:
                reason = {401: "API key not accepted", 400: "request rejected (number not verified?)"}.get(e.code, "")
                raise RuntimeError(f"CircuitDigest replied {e.code} {reason}".strip())
            except urllib.error.URLError as e:
                raise RuntimeError(f"CircuitDigest not reachable ({e.reason})")


def deliver_sms(text, messages=None, numbers=None):
    numbers = numbers or SMS_TO
    if SMS_BACKEND == "circuitdigest":
        send_via_circuitdigest(numbers, messages)
    elif SMS_BACKEND == "gsm":
        send_via_gsm(numbers, text)
    else:
        send_via_android(numbers, text)


def send_alert(m, kind, text=None, vibration=None, messages=None, alert_id=None):
    """Send one alert SMS to every configured number. Call without the lock held."""
    now = time.time()

    if not alerts_enabled:
        return False

    with lock:
        if m is not None:
            if now - m.last_alerts.get(kind, 0) < ALERT_COOLDOWN:
                alert_state["status"] = "COOLDOWN"
                m.add_event("alert", "SMS skipped", f"{ALERT_TITLES[kind]}: cooldown active")
                if alert_id:
                    store.set_alert(alert_id, sms="COOLDOWN")
                return False
            m.last_alerts[kind] = now
            text = alert_text(m, kind, vibration)
            messages = cd_messages(m, kind, vibration)

    try:
        deliver_sms(text, messages)
    except Exception as e:
        with lock:
            alert_state["status"] = "FAILED"
            if m is not None:
                m.last_alerts[kind] = 0
                m.add_event("critical", "SMS failed", str(e)[:120])
            if alert_id:
                store.set_alert(alert_id, sms="FAILED")
        print("SMS FAILED:", e)
        return False

    with lock:
        alert_state["status"] = "SENT"
        alert_state["sent"] += 1
        alert_state["lastSent"] = now_str()
        if m is not None:
            m.add_event("alert", "SMS alert sent", f"{ALERT_TITLES[kind]} to {len(SMS_TO)} number(s)")
        if alert_id:
            store.set_alert(alert_id, sms="SENT")

    print(f"SMS sent ({kind}) to {len(SMS_TO)} number(s) via {SMS_BACKEND}")
    return True


def record_alert(m, kind, vibration=None):
    """Log the alert for acknowledgement. Repeats while it is open only raise its count."""
    vibration = m.vibration if vibration is None else vibration
    detail = f"{vibration:.2f} g, motor {m.motor.lower()}"
    if m.health is not None:
        detail += f", health {m.health:.0f} %"
    alert_id, new = store.open_alert(m.id, kind, ALERT_TITLES[kind], detail)
    if m.simulated:
        store.set_alert(alert_id, sms="SIMULATED")
    if new:
        m.add_event("alert", f"Alert #{alert_id} opened", f"{ALERT_TITLES[kind]}, waiting for acknowledgement")
    return alert_id


def raise_alert(m, kind):
    """Record an alert and send its SMS from a background thread (used while the lock is held)."""
    alert_id = record_alert(m, kind)
    if m.simulated:
        store.set_alert(alert_id, sms="SIMULATED")
        return
    if not alerts_enabled:
        store.set_alert(alert_id, sms="DISABLED")
        return
    threading.Thread(target=send_alert, args=(m, kind), kwargs={"alert_id": alert_id}, daemon=True).start()


def handle_alert(m, vibration, pwm, status, motor=None, allow_alert=True):
    with lock:
        if time.time() - m.last_seen > 0.5:
            m.ingest(vibration, pwm, status, motor)
        m.alerts += 1
        m.add_event("critical", "Critical alert received", f"{m.id} escalated to fog")

        if status != "CRITICAL":
            return False

        alert_id = record_alert(m, "critical", vibration)
        if not allow_alert:
            m.add_event("alert", "Alert skipped", "Simulated machine, no message sent")
            store.set_alert(alert_id, sms="SIMULATED")
            return False
        if not alerts_enabled:
            store.set_alert(alert_id, sms="DISABLED")
            return False

    return send_alert(m, "critical", vibration=vibration, alert_id=alert_id)


def escalation_text(m, a):
    minutes = max(1, round((time.time() - a["t"]) / 60))
    lines = [
        "ResQFog ESCALATION: alert not acknowledged",
        f"Pump: {a['pump']} ({m.site if m else ''})",
        f"Alert #{a['id']}: {a['title']} at {date_str(a['t'])}",
        f"Open for {minutes} min, repeated {a['count']}x",
    ]
    if m is not None and m.lat is not None:
        lines.append(f"Map: {maps_link(m.lat, m.lon)}")
    lines.append("Acknowledge it on the ResQFog dashboard.")
    return sms_safe("\n".join(lines))


def escalate_once(now=None):
    """Send unacknowledged alerts older than ESCALATE_AFTER to the supervisor numbers."""
    now = now or time.time()
    for a in store.unacknowledged(now - ESCALATE_AFTER):
        store.set_alert(a["id"], escalated=now)
        with lock:
            m = machines.get(a["pump"])
            simulated = m is None or m.simulated
            if m is not None:
                m.add_event("alert", f"Alert #{a['id']} escalated",
                            "Not acknowledged in time" + (" (simulated, no SMS)" if simulated or not alerts_enabled
                                                          else ", supervisor notified"))
            text = escalation_text(m, a)
        if simulated or not alerts_enabled:
            continue
        pump = cd_field(f"pump {a['pump']}")
        try:
            deliver_sms(text, [(113, pump, cd_field(f"unacknowledged {a['kind']} alert"))],
                        numbers=SMS_ESCALATE_TO or SMS_TO)
            print(f"Escalation SMS sent for alert #{a['id']}")
        except Exception as e:
            with lock:
                if m is not None:
                    m.add_event("critical", "Escalation SMS failed", str(e)[:120])
            print("Escalation SMS FAILED:", e)


def background_jobs():
    """Every few seconds: health points, offline checks and escalation."""
    last_flush = time.time()
    while True:
        time.sleep(2)
        now = time.time()
        if now - last_flush >= HEALTH_INTERVAL:
            last_flush = now
            with lock:
                for m in machines.values():
                    m.flush_interval(now)
        try:
            escalate_once(now)
        except Exception as e:     # never let the job thread die
            print("Escalation check failed:", e)


# ---------------------------------------------------------------- routes

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        warning=WARNING_THRESHOLD,
        critical=CRITICAL_THRESHOLD,
        trip_after=TRIP_AFTER,
        watch=WATCH_BELOW,
        poor=POOR_BELOW,
        smooth=HEALTH_SMOOTH,
        sustain=HEALTH_SUSTAIN,
    )


@app.route("/api/status")
def api_status():
    now = time.time()
    wanted = request.args.get("machine", "").upper()

    with lock:
        for m in machines.values():
            m.check_timeout(now)

        fleet = visible_machines()
        selected = machines.get(wanted) if wanted in machines else None
        if selected is None or selected not in fleet:
            selected = (
                next((m for m in fleet if m.state in ("CRITICAL", "TRIPPED")), None)
                or next((m for m in fleet if not m.simulated), fleet[0])
            )

        detail = selected.detail(now)

        if alert_state["status"] == "COOLDOWN" and all(
            now - t >= ALERT_COOLDOWN for m in machines.values() for t in m.last_alerts.values()
        ):
            alert_state["status"] = "READY"

        return jsonify({
            "mode": "DEMO" if demo_mode else "LIVE",
            "serverTime": now_str(now),
            "uptime": format_duration(now - server_start_time),
            "thresholds": {"warning": WARNING_THRESHOLD, "critical": CRITICAL_THRESHOLD},
            "alerts": {
                "enabled": alerts_enabled,
                "channel": {"circuitdigest": "SMS (CircuitDigest)", "gsm": "SMS (GSM module)"}.get(SMS_BACKEND, "SMS (Android phone)"),
                "recipients": len(SMS_TO),
                "status": alert_state["status"],
                "sent": alert_state["sent"],
                "lastSent": alert_state["lastSent"],
                "cooldownRemaining": detail["machine"]["alertCooldown"],
                "escalateAfter": ESCALATE_AFTER,
                "escalateTo": len(SMS_ESCALATE_TO or SMS_TO),
            },
            "openAlertsTotal": len(store.alerts(open_only=True, limit=500)),
            "fleet": [m.summary(now) for m in fleet],
            **detail,
        })


@app.route("/faults/<path:filename>")
def download_fault(filename):
    folder = DEMO_FAULT_DIR if os.path.exists(os.path.join(DEMO_FAULT_DIR, filename)) else FAULT_DIR
    return send_from_directory(folder, filename, as_attachment=True, mimetype="text/csv")


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    with lock:
        m = machines.get(str(data.get("machine", "")).upper())
        if m is None:
            return jsonify({"status": "ERROR", "error": "unknown machine"}), 404
        m.reset_stats()
        m.history.clear()
        m.arrivals.clear()
        m.recovered = 0
        m.add_event("info", "Session reset", "Statistics cleared from dashboard")
    return jsonify({"status": "RESET"})


@app.route("/api/command", methods=["POST"])
def api_command():
    """Queue a command for an edge node. It is delivered in the reply to its next /data post."""
    data = request.get_json(silent=True) or {}
    command = str(data.get("command", "")).upper()

    if command != "RESET_TRIP":
        return jsonify({"status": "ERROR", "error": "unsupported command"}), 400

    with lock:
        m = machines.get(str(data.get("machine", "")).upper())
        if m is None:
            return jsonify({"status": "ERROR", "error": "unknown machine"}), 404
        if m.motor != "TRIPPED":
            return jsonify({"status": "ERROR", "error": "motor is not tripped"}), 409
        m.pending_command = command
        m.add_event("info", "Restart requested", "Sent to edge node with next reply")

    return jsonify({"status": "QUEUED"})


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    """Fit the ML model to this pump's own normal running (about 2 minutes)."""
    data = request.get_json(silent=True) or {}
    with lock:
        m = machines.get(str(data.get("machine", "")).upper())
        if m is None:
            return jsonify({"status": "ERROR", "error": "unknown machine"}), 404
        if m.simulated:
            return jsonify({"status": "ERROR", "error": "simulated pumps use the CWRU model"}), 409
        m.start_calibration(long=data.get("mode") == "long")
    return jsonify({"status": "CALIBRATING", "windows": CALIBRATION_WINDOWS})


def clean_text(value, limit=200):
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def optional_float(value, low, high):
    if value in (None, ""):
        return None
    v = float(value)
    if not low <= v <= high:
        raise ValueError(f"{v} is out of range")
    return v


@app.route("/api/pumps", methods=["GET"])
def api_pumps():
    """Registry and live state of every pump, for the map and the registry table."""
    now = time.time()
    with lock:
        live = {m.id: m.summary(now) for m in visible_machines()}
        rows = []
        for m in visible_machines():
            rows.append(live[m.id] | {"motorKw": m.motor_kw, "installed": m.installed, "notes": m.notes,
                                      "hours": m.hours(), "openAlerts": len(store.alerts(m.id, open_only=True))})
    return jsonify({"pumps": rows})


@app.route("/api/pumps", methods=["POST"])
def api_save_pump():
    data = request.get_json(silent=True) or {}
    pump_id = clean_text(data.get("id"), 16).upper()
    if not re.fullmatch(r"[A-Z0-9_-]{1,16}", pump_id):
        return jsonify({"status": "ERROR", "error": "pump ID: letters, digits, - and _ only"}), 400
    try:
        lat = optional_float(data.get("lat"), -90, 90)
        lon = optional_float(data.get("lon"), -180, 180)
        motor_kw = optional_float(data.get("motorKw"), 0, 1000)
    except (TypeError, ValueError) as e:
        return jsonify({"status": "ERROR", "error": f"invalid number: {e}"}), 400
    if (lat is None) != (lon is None):
        return jsonify({"status": "ERROR", "error": "give both latitude and longitude"}), 400

    row = store.save_pump(pump_id, site=clean_text(data.get("site"), 40) or None, lat=lat, lon=lon,
                          motor_kw=motor_kw, installed=clean_text(data.get("installed"), 20),
                          notes=clean_text(data.get("notes")))
    with lock:
        m = get_machine(pump_id)
        m.load_registry(hours=False)
        if lat is None:
            m.lat = m.lon = None
        m.add_event("info", "Pump details updated", row["site"] or pump_id)
    return jsonify({"status": "SAVED"})


@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    pump = request.args.get("machine", "").upper() or None
    open_only = request.args.get("open") == "1"
    return jsonify({"alerts": [alert_row(a) for a in store.alerts(pump, open_only=open_only, limit=100)]})


@app.route("/api/alerts/<int:alert_id>/ack", methods=["POST"])
def api_ack(alert_id):
    data = request.get_json(silent=True) or {}
    by = clean_text(data.get("by"), 40)
    if not by:
        return jsonify({"status": "ERROR", "error": "enter your name"}), 400
    a = store.alert(alert_id)
    if a is None:
        return jsonify({"status": "ERROR", "error": "unknown alert"}), 404
    if a["ack_t"]:
        return jsonify({"status": "ERROR", "error": "already acknowledged"}), 409
    store.set_alert(alert_id, ack_by=by, ack_t=time.time(), ack_note=clean_text(data.get("note")))
    with lock:
        m = machines.get(a["pump"])
        if m is not None:
            m.add_event("info", f"Alert #{alert_id} acknowledged", f"by {by}")
    return jsonify({"status": "ACKNOWLEDGED"})


@app.route("/api/maintenance", methods=["POST"])
def api_maintenance():
    data = request.get_json(silent=True) or {}
    technician = clean_text(data.get("technician"), 40)
    action = clean_text(data.get("action"), 80)
    if not technician or not action:
        return jsonify({"status": "ERROR", "error": "technician and work done are required"}), 400
    service = bool(data.get("service"))
    with lock:
        m = machines.get(str(data.get("machine", "")).upper())
        if m is None:
            return jsonify({"status": "ERROR", "error": "unknown machine"}), 404
        store.add_maintenance(m.id, technician, action, clean_text(data.get("notes"), 500),
                              m.run_seconds / 3600, service)
        if service:
            m.service_seconds = m.run_seconds
            m.serviced = True
            store.save_hours(m.id, m.run_seconds, m.service_seconds)
        m.add_event("info", "Maintenance logged", f"{action} by {technician}" + (" (service)" if service else ""))
    return jsonify({"status": "SAVED"})


@app.route("/report/<pump_id>")
def report(pump_id):
    """Printable maintenance report for one pump (use the browser's Print to PDF)."""
    now = time.time()
    with lock:
        m = machines.get(pump_id.upper())
        if m is None:
            abort(404)
        info = {
            "id": m.id, "site": m.site, "lat": m.lat, "lon": m.lon, "motorKw": m.motor_kw,
            "installed": m.installed, "notes": m.notes, "state": m.state, "health": m.health_summary(now),
            "hours": m.hours(), "ml": m.ml_summary(), "faults": list(m.captures)[:10], "simulated": m.simulated,
        }
    days = store.points(m.id, now - 7 * 86400, limit=20000)
    return render_template(
        "report.html", pump=info, generated=datetime.now().strftime("%d %b %Y %H:%M"),
        alerts=[alert_row(a) for a in store.alerts(m.id, limit=100)],
        maintenance=[maintenance_row(r) for r in store.maintenance(m.id, limit=100)],
        points=[{"t": datetime.fromtimestamp(p["t"]).strftime("%d %b %H:%M"), "h": p["health"],
                 "v": p["vib_avg"], "vmax": p["vib_max"]} for p in days[-2000:]],
        maps=maps_link(m.lat, m.lon) if m.lat is not None else None,
        warning=WARNING_THRESHOLD, critical=CRITICAL_THRESHOLD,
    )


@app.route("/report/<pump_id>.csv")
def report_csv(pump_id):
    """Every stored summary point of one pump (one row per interval)."""
    pump_id = pump_id.upper()
    rows = store.points(pump_id, 0, limit=1_000_000)
    lines = ["time,vibration_avg_g,vibration_max_g,health_pct,ml_score,run_hours"]
    for p in rows:
        fmt = lambda v, d: "" if v is None else f"{v:.{d}f}"
        lines.append(",".join([datetime.fromtimestamp(p["t"]).strftime("%Y-%m-%d %H:%M:%S"),
                               fmt(p["vib_avg"], 4), fmt(p["vib_max"], 4), fmt(p["health"], 1),
                               fmt(p["ml_score"], 4), fmt((p["run_seconds"] or 0) / 3600, 2)]))
    return Response("\n".join(lines) + "\n", mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=resqfog_{pump_id}_history.csv"})


def take_command(m):
    command, m.pending_command = m.pending_command, None
    return command or "NONE"


@app.route("/data", methods=["POST"])
def receive_data():
    try:
        data = request.get_json(force=True)
        vibration = float(data.get("vibration", 0))
        pwm = float(data.get("motorSpeed", 0))
    except Exception as e:
        print("Telemetry error:", e)
        return jsonify({"status": "ERROR", "error": str(e)}), 400

    with lock:
        m = get_machine(data.get("id"))
        m.update_site(data)
        m.ingest(vibration, pwm, data.get("status", ""), data.get("motor"), parse_features(data))
        command = take_command(m)

    return jsonify({"status": "RECEIVED", "fog": "ONLINE", "command": command})


@app.route("/data/batch", methods=["POST"])
def receive_batch():
    try:
        data = request.get_json(force=True)
        readings = list(data.get("readings", []))
    except Exception as e:
        print("Batch error:", e)
        return jsonify({"status": "ERROR", "error": str(e)}), 400

    with lock:
        m = get_machine(data.get("id"))
        m.update_site(data)
        count = m.ingest_backlog(readings)
        command = take_command(m)

    print(f"Recovered {count} buffered readings from {m.id}")
    return jsonify({"status": "RECEIVED", "accepted": count, "command": command})


@app.route("/alert", methods=["POST"])
def alert():
    try:
        data = request.get_json(force=True)
        vibration = float(data.get("vibration", 0))
        pwm = float(data.get("motorSpeed", 0))
    except Exception as e:
        print("Alert error:", e)
        return jsonify({"status": "ERROR", "error": str(e)}), 400

    print("CRITICAL ALERT RECEIVED:", data)

    with lock:
        m = get_machine(data.get("id"))
        m.update_site(data)

    status = str(data.get("status", "")).upper()
    sent = handle_alert(m, vibration, pwm, status, data.get("motor"))

    return jsonify({
        "status": "CRITICAL",
        "fog": "RECEIVED",
        "alert": "SENT" if sent else "NOT_SENT",
    })


CD_TEST = [(101, "ResQFog fog server", "online test SMS OK")]


def test_sms_text():
    return f"ResQFog test SMS: alerts from the fog server reach this number. {datetime.now().strftime('%d-%b %H:%M:%S')}"


@app.route("/test-sms", methods=["GET", "POST"])
def test_sms():
    if not alerts_enabled:
        return jsonify({"status": "NOT_SENT", "reason": "SMS not configured in .env"}), 500
    if send_alert(None, "test", test_sms_text(), messages=CD_TEST):
        return jsonify({"status": "SENT"})
    return jsonify({"status": "NOT_SENT", "reason": alert_state["status"]}), 500


# ---------------------------------------------------------------- simulator

def load_replay():
    try:
        data = np.load(os.path.join(BASE_DIR, "ml", "replay.npz"))
        return data["normal"], data["fault"]
    except (OSError, KeyError):
        return None, None


REPLAY_NORMAL, REPLAY_FAULT = load_replay()


DEMO_WEAR_START = 60          # s after start before the worn pump starts to degrade
DEMO_WEAR_SECONDS = 900       # s from new to fully worn


class SimulatedPump:
    """Imitates the ESP32 firmware for one pump: readings, alerts and the motor trip.

    The vibration window sent with each reading is a real CWRU recording:
    a healthy bearing normally, and a faulty one from 15 s before the
    threshold alarm, so the ML model can be seen warning early.

    One pump (wearing=True) instead degrades slowly: its windows blend from
    healthy to faulty recordings over 15 minutes, so its health score and the
    time-to-threshold estimate can be followed. Logging a service on the
    dashboard replaces the "bearing" and it starts again from new.
    """

    def __init__(self, machine, phase, fault_every, fault_offset, wearing=False):
        self.m = machine
        self.phase = phase
        self.fault_every = fault_every
        self.fault_offset = fault_offset
        self.wearing = wearing
        self.wear_start = DEMO_WEAR_START
        self.critical_count = 0
        self.tripped_at = None
        self.previous = "NORMAL"

    def wear(self, t):
        if not self.wearing:
            return 0.0
        return min(1.0, max(0.0, (t - self.wear_start) / DEMO_WEAR_SECONDS))

    def step(self, t):
        m = self.m

        if m.serviced:
            m.serviced = False
            self.wear_start = t + DEMO_WEAR_START

        if self.tripped_at is not None:
            # Waiting for a restart from the dashboard; a technician resets it after 30 s anyway
            if m.pending_command == "RESET_TRIP" or t - self.tripped_at > 30:
                m.pending_command = None
                self.tripped_at = None
                self.critical_count = 0
            else:
                m.ingest(random.uniform(0.01, 0.04), 0, "NORMAL", "TRIPPED")
                return

        fault_developing = not self.wearing and 25 <= (t + self.fault_offset) % self.fault_every < 53
        blend = self.wear(t) ** 2

        pwm = int(160 + 80 * math.sin((t + self.phase) / 25))

        # Normal running grows with speed and stays well under the warning level
        vibration = WARNING_THRESHOLD * (0.15 + pwm / 255 * 0.45) + random.uniform(-0.05, 0.05)

        cycle = (t + self.fault_offset) % self.fault_every
        if self.wearing:
            vibration += 0.85 * blend
        elif 40 <= cycle < 48:
            # developing fault: climbs through the WARNING band
            vibration = WARNING_THRESHOLD + (cycle - 40) / 8 * (CRITICAL_THRESHOLD - WARNING_THRESHOLD) \
                + random.uniform(-0.03, 0.03)
        elif 48 <= cycle < 53:
            # severe: stays CRITICAL long enough for the edge to trip the motor
            vibration = CRITICAL_THRESHOLD + random.uniform(0.03, 0.3)

        vibration = max(0.02, vibration)
        status = classify(vibration)

        self.critical_count = self.critical_count + 1 if status == "CRITICAL" else 0
        motor = "RUNNING"
        if self.critical_count >= TRIP_AFTER:
            motor = "TRIPPED"
            self.tripped_at = t

        window = None
        if REPLAY_NORMAL is not None:
            source = REPLAY_FAULT if fault_developing else REPLAY_NORMAL
            window = source[random.randrange(len(source))]
            if blend > 0:
                fault = REPLAY_FAULT[random.randrange(len(REPLAY_FAULT))]
                window = (1 - blend) * window + blend * (1 + blend) * fault

        m.ingest(vibration, pwm, status, motor, window_features(window) if window is not None else None)

        if status == "CRITICAL" and self.previous != "CRITICAL":
            m.alerts += 1
            m.add_event("critical", "Critical alert received", f"{m.id} escalated to fog")
            record_alert(m, "critical", vibration)

        self.previous = status


def run_simulator(pumps):
    t = 0
    while True:
        t += 1
        with lock:
            for pump in pumps:
                pump.step(t)
        time.sleep(1)


def seed_registry(sites):
    """Give the demo pumps a site and location unless the registry already has them."""
    for machine_id, site, lat, lon in sites:
        row = store.pump(machine_id)
        if row is None or not row["site"]:
            store.save_pump(machine_id, site=site, lat=lat, lon=lon, installed="2026-07-01")


def create_simulated_pumps(skip_first):
    # Different fault cycles so the pumps do not all fail together; PUMP-02 wears out slowly
    schedules = [(0, 140, 30), (17, 170, 95), (41, 110, 10), (63, 200, 150)]
    pumps = []
    for (machine_id, site, _, _), (phase, every, offset) in zip(SIM_SITES, schedules):
        if skip_first and machine_id == DEFAULT_MACHINE:
            continue
        m = Machine(machine_id, site, simulated=True)
        machines[machine_id] = m
        pumps.append(SimulatedPump(m, phase, every, offset, wearing=machine_id == "PUMP-02"))
    return pumps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResQFog fog server")
    parser.add_argument("--demo", action="store_true",
                        help="simulate four pumps (no hardware, no alerts)")
    parser.add_argument("--fleet", action="store_true",
                        help="real ESP32 as PUMP-01 plus three simulated pumps")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--db", default=DB_PATH, help="SQLite file for the registry, history and alerts")
    parser.add_argument("--test-sms", action="store_true",
                        help="send one test SMS with the settings in .env, then exit")
    args = parser.parse_args()

    if args.test_sms:
        if not alerts_enabled:
            print("SMS not configured in .env (see .env.example)")
            sys.exit(1)
        sys.exit(0 if send_alert(None, "test", test_sms_text(), messages=CD_TEST) else 1)

    demo_mode = args.demo

    # The demo keeps nothing on disk and runs the health timing six times faster
    DB_PATH = args.db
    store = Store(":memory:" if args.demo else DB_PATH)
    if args.demo:
        HEALTH_INTERVAL, TREND_WINDOW = 10, 600
        if not os.getenv("ESCALATE_AFTER"):
            ESCALATE_AFTER = 120

    if args.demo or args.fleet:
        seed_registry(SIM_SITES[1:] if args.fleet else SIM_SITES)
        pumps = create_simulated_pumps(skip_first=args.fleet)
        threading.Thread(target=run_simulator, args=(pumps,), daemon=True).start()

    if not args.demo:
        # Shown as "waiting" until the real ESP32 sends its first reading
        machines[DEFAULT_MACHINE] = Machine(DEFAULT_MACHINE, SIM_SITES[0][1], placeholder=not args.fleet)
        # pumps added on the dashboard earlier
        for row in store.pumps():
            if row["id"] not in machines:
                machines[row["id"]] = Machine(row["id"], row["site"] or "")

    threading.Thread(target=background_jobs, daemon=True).start()
    load_saved_captures()

    print()
    print("================================")
    print("        RESQFOG FOG SERVER")
    print("================================")
    if args.demo:
        print("MODE: DEMO (4 simulated pumps, alerts disabled)")
    elif args.fleet:
        print("MODE: FLEET (real PUMP-01 + 3 simulated pumps)")
    else:
        print("MODE: LIVE")
    if alerts_enabled:
        via = {"circuitdigest": "CircuitDigest", "gsm": "GSM module"}.get(SMS_BACKEND, "Android phone")
        print(f"Alerts:     SMS via {via} to {len(SMS_TO)} number(s), {ALERT_COOLDOWN // 60} min cooldown")
    else:
        print("WARNING: SMS not configured in .env - alerts disabled")
    print(f"Thresholds: warning {WARNING_THRESHOLD:.2f} g, critical {CRITICAL_THRESHOLD:.2f} g")
    print("ML model:  ", "loaded (CWRU)" if base_model else "missing - run python3 ml/train.py")
    print()
    print(f"Dashboard:  http://127.0.0.1:{args.port}")
    print(f"Telemetry:  POST /data, /data/batch, /alert")
    print(f"Fault CSVs: {FAULT_DIR}")
    print(f"Database:   {'in memory (demo)' if args.demo else DB_PATH}")
    print(f"Escalation: after {ESCALATE_AFTER // 60} min to {len(SMS_ESCALATE_TO or SMS_TO)} number(s)")
    print("================================")
    print()

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
