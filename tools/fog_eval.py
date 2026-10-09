"""Measure what the fog tier saves compared with sending everything to the cloud.

    python3 tools/fog_eval.py

Starts a fog server on a spare port with a temporary database and no SMS,
then measures:

  1. data volume: the JSON body an edge node posts every second, what the fog
     keeps per pump per day, and what would leave the site for a cloud copy
  2. fog latency: round trip of one reading with its window (features, ML,
     health) from a client on the same computer, plus the Wi-Fi hop to the
     router measured with ping
  3. capacity: readings per second the fog server handles with many pumps
  4. cloud round trip: HTTPS requests to an AWS region in Mumbai (ap-south-1),
     the closest public cloud region, with and without a kept-open connection
     (timed with curl)

Writes tools/fog_eval.json. The numbers depend on the computer and network.
"""

import http.client
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
sys.path.insert(0, ROOT)
from store import Store  # noqa: E402

PORT = 5391
CLOUD_HOST = "dynamodb.ap-south-1.amazonaws.com"     # answers /ping, no account needed
REPLAY = np.load(os.path.join(ROOT, "ml", "replay.npz"))


def reading(pump, i, window=None):
    """The JSON body the Wi-Fi firmware posts each second.

    The node sends the acceleration magnitude in milli-g, which sits around
    1000 (1 g) with the vibration on top, so 1 g is added to the CWRU window.
    """
    w = window if window is not None else REPLAY["normal"][i % len(REPLAY["normal"])]
    body = {"id": pump, "site": "Eval site", "vibration": 0.31, "motorSpeed": 180, "status": "NORMAL",
            "motor": "RUNNING", "w": [int(round((1.0 + v) * 1000)) for v in w]}
    return json.dumps(body, separators=(",", ":")).encode()


def post(conn, path, body):
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    response.read()
    return response.status


def start_server(db):
    env = dict(os.environ, SMS_TO="", MANAGER_PHONE="", SMS_GATEWAY_URL="", CIRCUITDIGEST_API_KEY="", GSM_PORT="")
    proc = subprocess.Popen([sys.executable, "fog_server.py", "--port", str(PORT), "--db", db], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(150):         # up to 30 s; the first import of scikit-learn can be slow
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=1).read()
            return proc
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("fog server did not start")


def calibrate(pump):
    conn = http.client.HTTPConnection("127.0.0.1", PORT)
    post(conn, "/data", reading(pump, 0))
    post(conn, "/api/calibrate", json.dumps({"machine": pump}).encode())
    start = time.perf_counter()
    for i in range(121):
        post(conn, "/data", reading(pump, i))
    return time.perf_counter() - start


def data_volume():
    body = len(reading("PUMP-01", 0))
    # one stored point per minute: measure the database growth for a day
    path = os.path.join(tempfile.mkdtemp(), "day.db")
    db = Store(path)
    empty = os.path.getsize(path)
    for i in range(1440):
        db.add_point("PUMP-01", 1.7e9 + 60 * i, 0.31, 0.45, 97.5, 0.43, 3600.0 + i)
    db.db.execute("VACUUM")
    stored = os.path.getsize(path) - empty
    summary = json.dumps({"id": "PUMP-01", "t": 1760000000, "vavg": 0.312, "vmax": 0.452, "health": 97.5,
                          "ml": 0.431, "hours": 1234.5, "state": "NORMAL"}, separators=(",", ":"))
    return {
        "edge_body_bytes": body,
        "raw_per_pump_day_MB": round(body * 86400 / 1e6, 1),
        "fog_db_per_pump_day_kB": round(stored / 1e3, 1),
        "cloud_summary_bytes": len(summary),
        "cloud_summary_per_pump_day_kB": round(len(summary) * 1440 / 1e3, 1),
        "reduction_percent": round(100 * (1 - len(summary) * 1440 / (body * 86400)), 2),
    }


def fog_latency(n=300):
    pump = "EVAL-LAT"
    calibrate(pump)
    conn = http.client.HTTPConnection("127.0.0.1", PORT)
    times = []
    for i in range(n):
        body = reading(pump, i, REPLAY["fault"][i % len(REPLAY["fault"])] if i % 2 else None)
        start = time.perf_counter()
        post(conn, "/data", body)
        times.append((time.perf_counter() - start) * 1000)
    return {"requests": n, "median_ms": round(statistics.median(times), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2)}


def wifi_hop():
    try:
        gateway = re.search(r"gateway: (\S+)", subprocess.run(["route", "-n", "get", "default"],
                                                              capture_output=True, text=True).stdout).group(1)
        out = subprocess.run(["ping", "-c", "20", "-i", "0.2", gateway], capture_output=True, text=True).stdout
        times = [float(x) for x in re.findall(r"time=([\d.]+) ms", out)]
        return {"pings": len(times), "median_ms": round(statistics.median(times), 2),
                "p95_ms": round(float(np.percentile(times, 95)), 2)} if times else None
    except (AttributeError, OSError):
        return None


def capacity(pumps=20, seconds=15):
    for p in range(pumps):
        calibrate(f"EVAL-{p:02d}")
    count = [0] * pumps
    stop = time.time() + seconds

    def client(p):
        conn = http.client.HTTPConnection("127.0.0.1", PORT)
        i = 0
        while time.time() < stop:
            post(conn, "/data", reading(f"EVAL-{p:02d}", i))
            count[p] += 1
            i += 1

    threads = [threading.Thread(target=client, args=(p,)) for p in range(pumps)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rate = sum(count) / seconds
    return {"clients": pumps, "seconds": seconds, "readings_per_s": round(rate, 1),
            "pumps_at_1_per_s": int(rate)}


def curl_times(urls):
    """Per-request total times from curl (uses the system certificates); curl keeps
    the connection open between URLs given in one call."""
    args = [x for url in urls for x in ("-o", "/dev/null", url)]
    out = subprocess.run(["curl", "-s", "-w", "%{time_total}\\n", *args],
                         capture_output=True, text=True, timeout=120).stdout
    return [float(x) * 1000 for x in out.split()]


def cloud_rtt(n=30):
    url = f"https://{CLOUD_HOST}/ping"
    fresh = [t for _ in range(n) for t in curl_times([url])]
    kept = curl_times([url] * (n + 1))[1:]          # the first one opens the connection
    if len(fresh) < n or len(kept) < n:
        raise OSError("cloud endpoint not reachable")
    stats = lambda t: {"median_ms": round(statistics.median(t), 1), "p95_ms": round(float(np.percentile(t, 95)), 1)}
    return {"host": CLOUD_HOST, "requests": n, "new_connection": stats(fresh), "kept_open": stats(kept)}


def main():
    db = os.path.join(tempfile.mkdtemp(), "eval.db")
    results = {"measured": time.strftime("%Y-%m-%d %H:%M"), "data_volume": data_volume()}
    proc = start_server(db)
    try:
        results["calibration_upload_s"] = round(calibrate("EVAL-CAL"), 2)
        results["fog_round_trip"] = fog_latency()
        results["capacity"] = capacity()
    finally:
        proc.terminate()
        proc.wait()
    results["wifi_hop_to_router"] = wifi_hop()
    try:
        results["cloud_round_trip"] = cloud_rtt()
    except OSError as e:
        results["cloud_round_trip"] = f"not measured: {e}"
    with open(os.path.join(ROOT, "tools", "fog_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
