"""SQLite storage on the fog node: pump registry, health history, alerts and maintenance.

Live readings stay in memory (fog_server.py). What is kept here is what the
maintenance team needs later: one summary point per pump per interval, every
alert with who acknowledged it, and the maintenance log. A day of one pump is
about 1,440 rows instead of 86,400 raw readings.
"""

import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS pumps (
    id TEXT PRIMARY KEY,
    site TEXT,
    lat REAL,
    lon REAL,
    motor_kw REAL,
    installed TEXT,
    notes TEXT,
    run_seconds REAL DEFAULT 0,
    service_seconds REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS points (
    pump TEXT,
    t REAL,
    vib_avg REAL,
    vib_max REAL,
    health REAL,
    ml_score REAL,
    run_seconds REAL
);
CREATE INDEX IF NOT EXISTS points_pump_t ON points (pump, t);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pump TEXT,
    kind TEXT,
    title TEXT,
    detail TEXT,
    t REAL,
    last_t REAL,
    count INTEGER DEFAULT 1,
    sms TEXT,
    escalated REAL,
    ack_by TEXT,
    ack_t REAL,
    ack_note TEXT
);
CREATE INDEX IF NOT EXISTS alerts_pump ON alerts (pump, t);
CREATE TABLE IF NOT EXISTS maintenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pump TEXT,
    t REAL,
    technician TEXT,
    action TEXT,
    notes TEXT,
    run_hours REAL,
    service INTEGER
);
"""

PUMP_FIELDS = ("site", "lat", "lon", "motor_kw", "installed", "notes")


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    def _all(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def _run(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.lastrowid

    # pump registry

    def pump(self, pump_id):
        rows = self._all("SELECT * FROM pumps WHERE id = ?", (pump_id,))
        return rows[0] if rows else None

    def pumps(self):
        return self._all("SELECT * FROM pumps ORDER BY id")

    def save_pump(self, pump_id, **fields):
        fields = {k: v for k, v in fields.items() if k in PUMP_FIELDS}
        self._run("INSERT OR IGNORE INTO pumps (id) VALUES (?)", (pump_id,))
        if fields:
            sets = ", ".join(f"{k} = ?" for k in fields)
            self._run(f"UPDATE pumps SET {sets} WHERE id = ?", (*fields.values(), pump_id))
        return self.pump(pump_id)

    def save_hours(self, pump_id, run_seconds, service_seconds):
        self._run("UPDATE pumps SET run_seconds = ?, service_seconds = ? WHERE id = ?",
                  (run_seconds, service_seconds, pump_id))

    # health history

    def add_point(self, pump, t, vib_avg, vib_max, health, ml_score, run_seconds):
        self._run("INSERT INTO points VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (pump, t, vib_avg, vib_max, health, ml_score, run_seconds))

    def points(self, pump, since=0.0, limit=5000):
        rows = self._all("SELECT * FROM points WHERE pump = ? AND t >= ? ORDER BY t DESC LIMIT ?",
                         (pump, since, limit))
        return rows[::-1]

    # alerts: one open alert per pump and kind; repeats only raise its count

    def open_alert(self, pump, kind, title, detail, t=None):
        """Record an alert. Returns (alert id, True if it is new)."""
        t = t or time.time()
        with self.lock:
            row = self.db.execute(
                "SELECT id FROM alerts WHERE pump = ? AND kind = ? AND ack_t IS NULL", (pump, kind)).fetchone()
            if row:
                self.db.execute("UPDATE alerts SET count = count + 1, last_t = ?, detail = ? WHERE id = ?",
                                (t, detail, row["id"]))
                self.db.commit()
                return row["id"], False
            cur = self.db.execute(
                "INSERT INTO alerts (pump, kind, title, detail, t, last_t) VALUES (?, ?, ?, ?, ?, ?)",
                (pump, kind, title, detail, t, t))
            self.db.commit()
            return cur.lastrowid, True

    def set_alert(self, alert_id, **fields):
        allowed = {k: v for k, v in fields.items() if k in ("sms", "escalated", "ack_by", "ack_t", "ack_note")}
        if allowed:
            sets = ", ".join(f"{k} = ?" for k in allowed)
            self._run(f"UPDATE alerts SET {sets} WHERE id = ?", (*allowed.values(), alert_id))

    def alert(self, alert_id):
        rows = self._all("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        return rows[0] if rows else None

    def alerts(self, pump=None, open_only=False, limit=50):
        where, args = [], []
        if pump:
            where.append("pump = ?")
            args.append(pump)
        if open_only:
            where.append("ack_t IS NULL")
        sql = "SELECT * FROM alerts" + (" WHERE " + " AND ".join(where) if where else "")
        return self._all(sql + " ORDER BY t DESC LIMIT ?", (*args, limit))

    def unacknowledged(self, older_than):
        return self._all("SELECT * FROM alerts WHERE ack_t IS NULL AND escalated IS NULL AND t <= ?",
                         (older_than,))

    # maintenance log

    def add_maintenance(self, pump, technician, action, notes, run_hours, service, t=None):
        return self._run("INSERT INTO maintenance (pump, t, technician, action, notes, run_hours, service) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (pump, t or time.time(), technician, action, notes, run_hours, int(service)))

    def maintenance(self, pump=None, limit=50):
        if pump:
            return self._all("SELECT * FROM maintenance WHERE pump = ? ORDER BY t DESC LIMIT ?", (pump, limit))
        return self._all("SELECT * FROM maintenance ORDER BY t DESC LIMIT ?", (limit,))
