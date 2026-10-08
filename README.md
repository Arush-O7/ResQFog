# ResQFog

Edge–fog–cloud condition monitoring for water-supply pumping stations, with an ML anomaly model trained on real bearing data.

ResQFog is our project for BCSE313L (Fundamentals of Fog and Edge Computing) at VIT. An ESP32 with an MPU6050 accelerometer is mounted on each pump motor. Every second it records a short vibration window, raises a local alarm and trips the motor if the vibration stays critical, and sends the reading to a Flask "fog" server. The fog server runs an Isolation Forest on the vibration window to warn about abnormal patterns before the threshold is crossed, shows all pumps on one dashboard, records the readings around every fault, and sends an SMS through Twilio.

![Dashboard](docs/images/dashboard.png)

## Why pumping stations

Borewell pumps, sump pumps and the pumps that fill overhead tanks are usually spread out and nobody is stationed at them. Most mechanical faults (bearing wear, impeller imbalance, misalignment) show up first as a change in vibration, but at an unmanned site they are only noticed when the water stops. The network at these sites is also unreliable, so the protection runs on the ESP32 itself, and the fog and cloud add early warning and visibility on top.

## How detection works

**Layer 1 – thresholds on the ESP32** (works without any network)

| State | v = \|√(ax²+ay²+az²) − 1 g\| | Response |
|---|---|---|
| NORMAL | < 1.00 g | – |
| WARNING | 1.00 – 1.20 g | logged, amber on the dashboard |
| CRITICAL | ≥ 1.20 g | buzzer, SMS, fault recording |
| TRIPPED | 3 CRITICAL readings in a row | motor power cut; restart by turning the knob to zero or from the dashboard |

**Layer 2 – ML anomaly detection on the fog server**

- The ESP32 records 256 readings at 500 Hz (MPU6050 with its 184 Hz low-pass filter) and sends them with each reading.
- The fog server computes the energy in 10 frequency bands (0–250 Hz) and scores it with an Isolation Forest.
- An anomaly is raised when 3 of the last 5 windows are abnormal. If the vibration level is still normal at that point, the dashboard shows an early warning.

The model is trained on the [CWRU bearing dataset](https://engineering.case.edu/bearingdatacenter), after filtering and resampling it to what the MPU6050 can actually measure. On held-out data:

| Isolation Forest (trained on normal data only) | |
|---|---|
| Fault windows detected | 97.7 % |
| False alarms on normal windows | 0.7 % |
| Precision / F1 / ROC AUC | 0.999 / 0.988 / 0.999 |
| Random Forest reference, trained on 0–2 hp and tested on 3 hp | 87.4 % accuracy |

Full numbers are in [ml/results.json](ml/results.json). Shape features such as kurtosis and crest factor only reached about 15 % detection at this bandwidth, which is why the model uses band energies.

![Detection panel and event log](docs/images/detection_and_events.png)

Every pump is different, so a real pump is **calibrated** with the "Calibrate ML" button: it fits the same kind of model to about 2 minutes of that pump's normal running. Simulated pumps in demo mode replay real CWRU windows and use the CWRU model.

## Other features

- **Fleet dashboard** – card per pump, live graph, statistics, event log, system health; works on a phone.
- **SMS** – pump ID, site and reading, at most one SMS per minute per pump.
- **Offline buffer** – if the fog server can't be reached, the ESP32 keeps up to 300 readings (5 minutes) and uploads them later.
- **Fault recordings** – 50 readings before and 50 after every new fault, saved as CSV in `fault_logs/`.

## Hardware

| Component | Connection to ESP32 |
|---|---|
| MPU6050 accelerometer | SDA GPIO 21, SCL GPIO 22 (address 0x68) |
| 16×2 LCD with I²C backpack | same I²C bus (address 0x27) |
| Potentiometer (speed / trip reset) | GPIO 34 |
| L298N motor driver | ENA GPIO 25, IN1 GPIO 26, IN2 GPIO 27 |
| Buzzer | GPIO 32 |

About ₹1,050 per node with hobby modules.

## Setup

### Edge node

1. Open `edge/resqfog_edge/resqfog_edge.ino` in Arduino IDE (ESP32 board package installed).
2. Install **LiquidCrystal_I2C** from the Library Manager.
3. Copy `secrets.example.h` to `secrets.h` in the same folder and set the Wi-Fi name, password and `FOG_SERVER` (the IP of the laptop running the fog server; on macOS `ipconfig getifaddr en0`).
4. Set `MACHINE_ID` and `SITE_NAME` for the pump, upload, and open the Serial Monitor at 115200.

### Fog server

```bash
pip install -r requirements.txt
cp .env.example .env      # Twilio values (optional)
python3 fog_server.py
```

Open `http://127.0.0.1:5001`, or `http://<laptop-ip>:5001` from a phone on the same network. Then select the pump and press **Calibrate ML** while it runs normally (change the speed a little during the 2 minutes).

| Command | What it runs |
|---|---|
| `python3 fog_server.py` | Real edge nodes, real SMS |
| `python3 fog_server.py --fleet` | Real ESP32 as PUMP-01 plus three simulated pumps |
| `python3 fog_server.py --demo` | Four simulated pumps, no hardware, no SMS |

### Retraining the model

```bash
python3 ml/train.py
```

This downloads the CWRU files it needs (about 100 MB) into `ml/data/`, and writes `ml/model.joblib`, `ml/results.json` and `ml/replay.npz`.

`python3 ml/experiments.py` runs the extra experiments (feature ablation, detector comparison, 10 seeds, leave-one-load-out, 12 kHz vs MPU6050 bandwidth, timing) and writes `ml/experiments.json`.

The thresholds are in two places and must match: the top of `fog_server.py` and the sketch (1.00 g and 1.20 g).

## Fog server API

| Endpoint | Purpose |
|---|---|
| `POST /data` | One reading: id, site, vibration, motorSpeed, status, motor, and `w` (256 values in milli-g). The reply may contain `"command": "RESET_TRIP"` |
| `POST /data/batch` | Readings buffered during an outage, each with its age in ms |
| `POST /alert` | Critical alert, triggers the SMS |
| `GET /api/status?machine=PUMP-01` | All pumps plus details of one pump, including its ML state |
| `POST /api/calibrate` | `{"machine": "PUMP-01"}` – fit the ML model to this pump |
| `POST /api/command` | `{"machine": "PUMP-01", "command": "RESET_TRIP"}` |
| `POST /api/reset` | Clear statistics for one pump |
| `GET /faults/<file>` | Download a fault CSV |

## Project structure

```
fog_server.py               Flask fog server and pump simulator
templates/dashboard.html    dashboard
static/                     Chart.js, served locally
ml/features.py              band-energy features (used for training and live)
ml/train.py                 downloads CWRU data, trains and evaluates the model
ml/model.joblib             trained model
ml/results.json             evaluation results
edge/resqfog_edge/          ESP32 firmware
docs/                       presentation, screenshots, related work, sample data
```

## Limitations

- The MPU6050 bandwidth (184 Hz) cannot capture the high-frequency impacts of early bearing faults; the model relies on low-frequency band energy.
- In CWRU, healthy and faulty bearings were recorded separately, so part of the difference may come from the recordings themselves (see Smith and Randall, 2015).
- The model has not yet been tested on faults induced on our own rig. That is the next step, together with a current sensor for dry-running detection.

Comparison with existing work: [docs/related_work.md](docs/related_work.md).

## Team

Arush Mishra (23BCT0199) and Abuzar Siddiqi (23BCT0186)
Faculty: Prof. Kauser Ahmed P
