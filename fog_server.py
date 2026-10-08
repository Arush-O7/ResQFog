"""ResQFog fog server.

Receives telemetry from the ESP32 edge nodes installed on each pump, keeps
per-machine state for the dashboard, runs the ML anomaly model on the
vibration window that comes with each reading, records the readings around
every fault as CSV, and sends alerts to the maintenance team on Telegram.

Run:
    python3 fog_server.py                  # real edge nodes, real alerts
    python3 fog_server.py --fleet          # real PUMP-01 + three simulated pumps
    python3 fog_server.py --demo           # four simulated pumps, no hardware, no alerts
    python3 fog_server.py --find-chat-id   # show Telegram chat IDs (setup)
"""

import argparse
import csv
import html
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
from flask import Flask, jsonify, render_template, request, send_from_directory
from sklearn.ensemble import IsolationForest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "ml"))
from features import WINDOW, window_features, window_rms  # noqa: E402

load_dotenv()

# Alerts go to a Telegram bot (free, any text, can send a map pin).
# TELEGRAM_CHAT_ID may list several chats or groups, separated by commas.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_IDS = [c.strip() for c in os.getenv("TELEGRAM_CHAT_ID", "").split(",") if c.strip()]
TELEGRAM_API = os.getenv("TELEGRAM_API_BASE", "https://api.telegram.org").rstrip("/")

# Must match WARNING_THRESHOLD / CRITICAL_THRESHOLD in edge/resqfog_edge/resqfog_edge.ino
WARNING_THRESHOLD = 1.00
CRITICAL_THRESHOLD = 1.20

# Edge trips the motor after this many consecutive CRITICAL readings (same as TRIP_AFTER on the ESP32)
TRIP_AFTER = 3

ALERT_COOLDOWN = 60        # seconds between alerts of the same kind for the same pump
EDGE_TIMEOUT = 5           # seconds without telemetry before a node is shown offline

HISTORY_SIZE = 120
EVENT_SIZE = 50

FAULT_PRE_SAMPLES = 50
FAULT_POST_SAMPLES = 50
FAULT_LIST_SIZE = 20

DEFAULT_MACHINE = "PUMP-01"

FAULT_DIR = os.path.join(BASE_DIR, "fault_logs")
DEMO_FAULT_DIR = os.path.join(FAULT_DIR, "demo")   # simulated pumps, kept apart from real data

STATES = ("NORMAL", "WARNING", "CRITICAL")

# ML anomaly detection (see ml/train.py)
ML_MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.joblib")
ML_PUMP_DIR = os.path.join(BASE_DIR, "ml", "pumps")     # per-pump models from calibration
CALIBRATION_WINDOWS = 120     # about 2 minutes of normal running
ML_VOTES = 5                  # look at the last 5 windows...
ML_NEEDED = 3                 # ...and call it an anomaly if 3 of them are abnormal
ML_MIN_SPEED = 10             # % - below this the motor is treated as stopped

# Pumping stations used by the simulator (--demo and --fleet)
SIM_SITES = [
    ("PUMP-01", "VIT Main Sump"),
    ("PUMP-02", "Katpadi Pump House"),
    ("PUMP-03", "Gandhi Nagar OHT"),
    ("PUMP-04", "Sathuvachari Borewell"),
]


app = Flask(__name__)

alerts_enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS)

lock = threading.Lock()

server_start_time = time.time()

demo_mode = False

alert_state = {
    "status": "READY" if alerts_enabled else "DISABLED",
    "sent": 0,
    "lastSent": "--",
}

machines = {}

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

        # location (GPS fix or installed coordinates) and radio link, if the node sends them
        self.lat = None
        self.lon = None
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

        self.reset_stats()

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
        if data.get("site"):
            self.site = str(data["site"])[:40]

        try:
            lat, lon = float(data["lat"]), float(data["lon"])
            if lat or lon:
                fix = bool(int(data.get("gps", 0)))
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

    def start_calibration(self):
        self.ml_calibration = []
        self.ml_flags.clear()
        self.ml_state = "CALIBRATING"
        self.add_event("info", "ML calibration started", f"Collecting {CALIBRATION_WINDOWS} windows of normal running")

    def score_window(self, features):
        if self.motor == "TRIPPED" or self.speed_pct < ML_MIN_SPEED:
            if self.ml_state in ("NORMAL", "ANOMALY"):
                self.ml_state = "MOTOR STOPPED"
                self.ml_flags.clear()   # start fresh votes after a restart
            return
        if self.ml_state == "MOTOR STOPPED":
            self.ml_state = "NORMAL"


        if self.ml_calibration is not None:
            self.ml_calibration.append(features)
            if len(self.ml_calibration) >= CALIBRATION_WINDOWS:
                self.finish_calibration()
            return

        if not self.ml_model:
            return

        score = float(-self.ml_model["model"].score_samples([features])[0])
        self.ml_score = score
        self.ml_flags.append(score > self.ml_model["threshold"])

        new_state = "ANOMALY" if sum(self.ml_flags) >= ML_NEEDED else "NORMAL"
        if new_state != self.ml_state:
            if new_state == "ANOMALY":
                detail = "Vibration pattern differs from normal running"
                if self.status == "NORMAL":
                    detail += ", before the threshold alarm"
                self.add_event("warning", "ML anomaly detected", detail)
                alert_later(self, "ml")
            elif self.ml_state == "ANOMALY":
                self.add_event("normal", "ML pattern back to normal")
        self.ml_state = new_state

    def finish_calibration(self):
        data = np.array(self.ml_calibration)
        model = IsolationForest(n_estimators=100, random_state=0).fit(data)
        # 95th percentile per window; the 3-of-5 vote keeps false alarms near 0.1 %
        threshold = float(np.quantile(-model.score_samples(data), 0.95))
        self.ml_model = {"model": model, "threshold": threshold, "source": f"{self.id} calibration"}
        self.ml_calibration = None
        self.ml_state = "NORMAL"
        try:
            os.makedirs(ML_PUMP_DIR, exist_ok=True)
            joblib.dump(self.ml_model, os.path.join(ML_PUMP_DIR, f"{self.id}.joblib"))
        except OSError as e:
            print("Could not save pump model:", e)
        self.add_event("info", "ML calibration finished", f"Model fitted on {len(data)} windows")

    def ml_summary(self):
        return {
            "state": self.ml_state,
            "score": round(self.ml_score, 3) if self.ml_score is not None else None,
            "threshold": round(self.ml_model["threshold"], 3) if self.ml_model else None,
            "model": ("CWRU bearing data" if self.ml_model.get("source") == "CWRU" else "calibrated on this pump")
            if self.ml_model else None,
            "progress": len(self.ml_calibration) if self.ml_calibration is not None else None,
            "needed": CALIBRATION_WINDOWS,
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
                alert_later(self, "tripped")
            else:
                self.add_event("normal", "Motor restarted", "Trip cleared")
            self.motor = motor

        self.record_stats(vibration, now, status)

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


# ---------------------------------------------------------------- alerts (Telegram)

ALERT_TITLES = {
    "critical": "CRITICAL vibration",
    "tripped": "Motor TRIPPED by edge protection",
    "ml": "Early warning: abnormal vibration pattern",
}

ALERT_ACTIONS = {
    "critical": "Inspect the pump. The ESP32 trips the motor if this lasts 3 readings.",
    "tripped": "Inspect the pump, then restart it from the dashboard or with the speed knob.",
    "ml": "Vibration level is still normal. Plan an inspection soon.",
}


def telegram(method, payload):
    """POST one Telegram Bot API call; raises on any failure."""
    request = urllib.request.Request(
        f"{TELEGRAM_API}/bot{TELEGRAM_BOT_TOKEN}/{method}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            reply = json.loads(response.read())
    except urllib.error.HTTPError as e:
        reply = json.loads(e.read() or b"{}")
    if not reply.get("ok"):
        raise RuntimeError(reply.get("description", "Telegram request failed"))
    return reply["result"]


def alert_text(m, kind, vibration=None):
    """The message the maintenance team receives (Telegram HTML formatting)."""
    e = html.escape
    vibration = m.vibration if vibration is None else vibration
    lines = [
        f"<b>ResQFog ALERT: {ALERT_TITLES[kind]}</b>",
        "",
        f"<b>Pump:</b> {e(m.id)} ({e(m.site)})",
        f"<b>Vibration:</b> {vibration:.2f} g (warning {WARNING_THRESHOLD:.2f} g, critical {CRITICAL_THRESHOLD:.2f} g)",
        f"<b>Motor:</b> {'TRIPPED' if m.motor == 'TRIPPED' else f'running at {m.speed_pct} %'}",
    ]
    if m.ml_score is not None and m.ml_model:
        lines.append(f"<b>ML check:</b> {m.ml_state.lower()} (score {m.ml_score:.3f}, limit {m.ml_model['threshold']:.3f})")
    lines.append(f"<b>Time:</b> {datetime.now().strftime('%d %b %Y, %H:%M:%S')}")
    if m.lat is not None:
        source = f"GPS fix, {m.satellites} satellites" if m.gps_fix else "installed location"
        lines.append(f"<b>Location:</b> {m.lat:.5f}, {m.lon:.5f} ({source})")
        lines.append(f'<b>Map:</b> <a href="{maps_link(m.lat, m.lon)}">open in Google Maps</a>')
    if m.link == "LoRa" and m.rssi is not None:
        lines.append(f"<b>Link:</b> LoRa, RSSI {m.rssi} dBm, SNR {m.snr} dB")
    lines += ["", f"<b>Action:</b> {ALERT_ACTIONS[kind]}"]
    return "\n".join(lines)


def send_alert(m, kind, text=None, vibration=None):
    """Send one alert to every configured chat. Call without the lock held."""
    now = time.time()

    if not alerts_enabled:
        return False

    with lock:
        if m is not None:
            if now - m.last_alerts.get(kind, 0) < ALERT_COOLDOWN:
                alert_state["status"] = "COOLDOWN"
                m.add_event("alert", "Alert skipped", f"{ALERT_TITLES[kind]}: cooldown active")
                return False
            m.last_alerts[kind] = now
            text = alert_text(m, kind, vibration)
            location = (m.lat, m.lon) if m.lat is not None and kind != "ml" else None
        else:
            location = None

    try:
        for chat in TELEGRAM_CHAT_IDS:
            telegram("sendMessage", {"chat_id": chat, "text": text, "parse_mode": "HTML",
                                     "disable_web_page_preview": True})
            if location:
                telegram("sendLocation", {"chat_id": chat, "latitude": location[0], "longitude": location[1]})
    except Exception as e:
        with lock:
            alert_state["status"] = "FAILED"
            if m is not None:
                m.last_alerts[kind] = 0
                m.add_event("critical", "Alert failed", str(e)[:120])
        print("ALERT FAILED:", e)
        return False

    with lock:
        alert_state["status"] = "SENT"
        alert_state["sent"] += 1
        alert_state["lastSent"] = now_str()
        if m is not None:
            m.add_event("alert", "Alert sent on Telegram", ALERT_TITLES[kind])

    print(f"Alert sent ({kind}) to {len(TELEGRAM_CHAT_IDS)} chat(s)")
    return True


def alert_later(m, kind):
    """Send an alert from a background thread (used while the lock is held)."""
    if m.simulated or not alerts_enabled:
        return
    threading.Thread(target=send_alert, args=(m, kind), daemon=True).start()


def handle_alert(m, vibration, pwm, status, motor=None, allow_alert=True):
    with lock:
        if time.time() - m.last_seen > 0.5:
            m.ingest(vibration, pwm, status, motor)
        m.alerts += 1
        m.add_event("critical", "Critical alert received", f"{m.id} escalated to fog")

        if status != "CRITICAL":
            return False

        if not allow_alert:
            m.add_event("alert", "Alert skipped", "Simulated machine, no message sent")
            return False

    return send_alert(m, "critical", vibration=vibration)


def find_chat_ids():
    """Print the chats that have messaged the bot, to fill in TELEGRAM_CHAT_ID."""
    if not TELEGRAM_BOT_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN in .env first (create the bot with @BotFather).")
        return
    updates = telegram("getUpdates", {})
    chats = {}
    for u in updates:
        chat = (u.get("message") or u.get("channel_post") or u.get("my_chat_member") or {}).get("chat")
        if chat:
            chats[chat["id"]] = chat.get("title") or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")]))
    if not chats:
        print("No chats yet. Send any message to the bot (or add it to a group and post there), then run this again.")
    for chat_id, name in chats.items():
        print(f"{chat_id}\t{name}")


# ---------------------------------------------------------------- routes

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        warning=WARNING_THRESHOLD,
        critical=CRITICAL_THRESHOLD,
        trip_after=TRIP_AFTER,
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
                "channel": "Telegram",
                "chats": len(TELEGRAM_CHAT_IDS),
                "status": alert_state["status"],
                "sent": alert_state["sent"],
                "lastSent": alert_state["lastSent"],
                "cooldownRemaining": detail["machine"]["alertCooldown"],
            },
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
        m.start_calibration()
    return jsonify({"status": "CALIBRATING", "windows": CALIBRATION_WINDOWS})


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


@app.route("/test-alert", methods=["GET", "POST"])
def test_alert():
    if not alerts_enabled:
        return jsonify({"status": "NOT_SENT", "reason": "Telegram not configured in .env"}), 500
    text = (f"<b>ResQFog test message</b>\nAlerts from the fog server reach this chat.\n"
            f"<b>Time:</b> {datetime.now().strftime('%d %b %Y, %H:%M:%S')}")
    if send_alert(None, "test", text):
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


class SimulatedPump:
    """Imitates the ESP32 firmware for one pump: readings, alerts and the motor trip.

    The vibration window sent with each reading is a real CWRU recording:
    a healthy bearing normally, and a faulty one from 15 s before the
    threshold alarm, so the ML model can be seen warning early.
    """

    def __init__(self, machine, phase, fault_every, fault_offset):
        self.m = machine
        self.phase = phase
        self.fault_every = fault_every
        self.fault_offset = fault_offset
        self.critical_count = 0
        self.tripped_at = None
        self.previous = "NORMAL"

    def step(self, t):
        m = self.m

        if self.tripped_at is not None:
            # Waiting for a restart from the dashboard; a technician resets it after 30 s anyway
            if m.pending_command == "RESET_TRIP" or t - self.tripped_at > 30:
                m.pending_command = None
                self.tripped_at = None
                self.critical_count = 0
            else:
                m.ingest(random.uniform(0.01, 0.04), 0, "NORMAL", "TRIPPED")
                return

        fault_developing = 25 <= (t + self.fault_offset) % self.fault_every < 53

        pwm = int(160 + 80 * math.sin((t + self.phase) / 25))

        # Normal running grows with speed and stays well under the warning level
        vibration = WARNING_THRESHOLD * (0.15 + pwm / 255 * 0.45) + random.uniform(-0.05, 0.05)

        cycle = (t + self.fault_offset) % self.fault_every
        if 40 <= cycle < 48:
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

        m.ingest(vibration, pwm, status, motor, window_features(window) if window is not None else None)

        if status == "CRITICAL" and self.previous != "CRITICAL":
            m.alerts += 1
            m.add_event("critical", "Critical alert received", f"{m.id} escalated to fog")
            m.add_event("alert", "Alert skipped", "Simulated machine, no message sent")

        self.previous = status


def run_simulator(pumps):
    t = 0
    while True:
        t += 1
        with lock:
            for pump in pumps:
                pump.step(t)
        time.sleep(1)


def create_simulated_pumps(skip_first):
    # Different fault cycles so the pumps do not all fail together
    schedules = [(0, 140, 30), (17, 170, 95), (41, 110, 10), (63, 200, 150)]
    pumps = []
    for (machine_id, site), (phase, every, offset) in zip(SIM_SITES, schedules):
        if skip_first and machine_id == DEFAULT_MACHINE:
            continue
        m = Machine(machine_id, site, simulated=True)
        machines[machine_id] = m
        pumps.append(SimulatedPump(m, phase, every, offset))
    return pumps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResQFog fog server")
    parser.add_argument("--demo", action="store_true",
                        help="simulate four pumps (no hardware, no alerts)")
    parser.add_argument("--fleet", action="store_true",
                        help="real ESP32 as PUMP-01 plus three simulated pumps")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--find-chat-id", action="store_true",
                        help="list the Telegram chats that have messaged the bot, then exit")
    args = parser.parse_args()

    if args.find_chat_id:
        find_chat_ids()
        sys.exit(0)

    demo_mode = args.demo

    if args.demo or args.fleet:
        pumps = create_simulated_pumps(skip_first=args.fleet)
        threading.Thread(target=run_simulator, args=(pumps,), daemon=True).start()

    if not args.demo:
        # Shown as "waiting" until the real ESP32 sends its first reading
        machines[DEFAULT_MACHINE] = Machine(DEFAULT_MACHINE, SIM_SITES[0][1], placeholder=not args.fleet)

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
        print(f"Alerts:     Telegram, {len(TELEGRAM_CHAT_IDS)} chat(s)")
    else:
        print("WARNING: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in .env - alerts disabled")
    print(f"Thresholds: warning {WARNING_THRESHOLD:.2f} g, critical {CRITICAL_THRESHOLD:.2f} g")
    print("ML model:  ", "loaded (CWRU)" if base_model else "missing - run python3 ml/train.py")
    print()
    print(f"Dashboard:  http://127.0.0.1:{args.port}")
    print(f"Telemetry:  POST /data, /data/batch, /alert")
    print(f"Fault CSVs: {FAULT_DIR}")
    print("================================")
    print()

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
