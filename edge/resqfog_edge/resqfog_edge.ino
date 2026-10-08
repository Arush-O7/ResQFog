// ResQFog edge node - ESP32 + MPU6050 on a pump motor
//
// Every second: record a short vibration window, classify the vibration,
// drive the buzzer/LCD, and send the reading (with the window, for the ML
// model on the fog server). The alarm and the motor trip are decided here
// on the ESP32, so they still work when the Wi-Fi or the laptop is down.
// Readings taken while the fog server is unreachable are kept in RAM and
// uploaded once it is back.
//
// Libraries: LiquidCrystal_I2C; TinyGPSPlus if USE_GPS; LoRa if USE_LORA

// ---------------- Optional modules (1 = on) ----------------

#define USE_GPS  0   // NEO-6M GPS on UART2, see gps_module.h
#define USE_LORA 0   // SX1278 LoRa to a gateway instead of Wi-Fi, see lora_link.h

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#if USE_GPS
#include "gps_module.h"
#endif

#if USE_LORA
#include "band_features.h"
#include "lora_link.h"
#endif

// ---------------- Wi-Fi and fog server ----------------

// WIFI_SSID, WIFI_PASSWORD and FOG_SERVER live in secrets.h
// (copy secrets.example.h to secrets.h the first time)
#include "secrets.h"

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;
const char* fogServer = FOG_SERVER;

// ---------------- Machine identity ----------------

#define MACHINE_ID "PUMP-01"
#define MACHINE_NUMBER 1          // used as the node number over LoRa
#define SITE_NAME  "VIT Main Sump"

// Installed location, used when there is no GPS or no GPS fix
const double SITE_LAT = 12.96920;
const double SITE_LON = 79.15590;

// Over LoRa a reading is sent every LORA_REPORT_EVERY seconds, and at once
// when the state changes. With one report per 30 s, about 20 pumps can share
// one gateway at SF7 with under 10 % packet collisions (pure ALOHA estimate).
const int LORA_REPORT_EVERY = 30;
const uint16_t LORA_REPLY_WAIT_MS = 300;

// ---------------- Pins ----------------

#define SDA_PIN 21
#define SCL_PIN 22
#define MPU_ADDR 0x68

#define POT_PIN 34

#define ENA_PIN 25
#define IN1_PIN 26
#define IN2_PIN 27

#define BUZZER_PIN 32

LiquidCrystal_I2C lcd(0x27, 16, 2);

// ---------------- Detection settings ----------------

// Must match WARNING_THRESHOLD / CRITICAL_THRESHOLD in fog_server.py
const float WARNING_THRESHOLD = 1.00;
const float CRITICAL_THRESHOLD = 1.20;

// Cut motor power after this many CRITICAL readings in a row
const int TRIP_AFTER = 3;

const unsigned long SAMPLE_INTERVAL = 1000;

// Vibration window for the ML model: 256 readings, one every 2 ms (500 Hz).
// Must match FS and WINDOW in ml/features.py.
const int WINDOW = 256;
const unsigned long WINDOW_STEP_US = 2000;
float windowData[WINDOW];
const uint16_t HTTP_TIMEOUT_MS = 1500;

// The fog server already limits alerts to one per minute per machine
const unsigned long ALERT_COOLDOWN = 0;

// ---------------- Offline buffer ----------------

struct Reading {
  float vibration;
  uint8_t pwm;
  uint8_t status;
  uint32_t takenAt;
};

const int BACKLOG_SIZE = 300;   // 5 minutes at one reading per second
const int BATCH_SIZE = 30;      // readings uploaded per request

Reading backlog[BACKLOG_SIZE];
int backlogStart = 0;
int backlogCount = 0;

const char* STATUS_NAMES[] = {"NORMAL", "WARNING", "CRITICAL"};

// ---------------- State ----------------

float ax = 0.0, ay = 0.0, az = 0.0;
float vibration = 0.0;

int motorSpeed = 0;
int speedPercent = 0;
int potPercent = 0;

String currentStatus = "NORMAL";

int criticalCount = 0;
bool motorTripped = false;

bool criticalSent = false;
unsigned long lastAlertTime = 0;

bool fogOnline = false;

unsigned long lastSampleTime = 0;
unsigned long lastWifiAttempt = 0;

#if USE_LORA
int readingsSinceReport = 0;
String lastReportedStatus = "";
bool lastReportedTrip = false;
bool loraReady = false;
#endif

// ---------------- MPU6050 ----------------

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void readAccelerometer() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 6);

  if (Wire.available() >= 6) {
    int16_t rawAx = ((int16_t)Wire.read() << 8) | Wire.read();
    int16_t rawAy = ((int16_t)Wire.read() << 8) | Wire.read();
    int16_t rawAz = ((int16_t)Wire.read() << 8) | Wire.read();

    // +-2 g range: 16384 counts per g
    ax = rawAx / 16384.0;
    ay = rawAy / 16384.0;
    az = rawAz / 16384.0;
  }
}

float magnitude() {
  return sqrt(ax * ax + ay * ay + az * az);
}

float calculateVibration() {
  return fabs(magnitude() - 1.0);   // remove gravity
}

// Fills windowData with the acceleration magnitude, one reading every 2 ms
void captureWindow() {
  unsigned long next = micros();
  for (int i = 0; i < WINDOW; i++) {
    while ((long)(micros() - next) < 0) {}
    next += WINDOW_STEP_US;
    readAccelerometer();
    windowData[i] = magnitude();
  }
}

String classifyVibration(float value) {
  if (value >= CRITICAL_THRESHOLD) return "CRITICAL";
  if (value >= WARNING_THRESHOLD) return "WARNING";
  return "NORMAL";
}

uint8_t statusCode(const String& status) {
  if (status == "CRITICAL") return 2;
  if (status == "WARNING") return 1;
  return 0;
}

// ---------------- Wi-Fi ----------------

void connectWiFi() {
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi");

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  lcd.clear();
  lcd.setCursor(0, 0);

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Wi-Fi connected. ESP32 IP: ");
    Serial.println(WiFi.localIP());
    lcd.print("WiFi Connected");
    lcd.setCursor(0, 1);
    lcd.print(WiFi.localIP());
  } else {
    Serial.println("Wi-Fi failed. Running offline, will retry.");
    lcd.print("WiFi Failed");
    lcd.setCursor(0, 1);
    lcd.print("Local mode");
  }

  lastWifiAttempt = millis();
  delay(1500);
}

void maintainWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  if (millis() - lastWifiAttempt > 10000) {
    Serial.println("Wi-Fi lost. Reconnecting...");
    WiFi.disconnect();
    WiFi.begin(ssid, password);
    lastWifiAttempt = millis();
  }
}

// ---------------- Motor protection ----------------

void tripMotor() {
  motorTripped = true;
  analogWrite(ENA_PIN, 0);

  Serial.println();
  Serial.println("!!! MOTOR TRIPPED - critical vibration for 3 readings !!!");
  Serial.println("Turn the speed knob to zero, or restart from the dashboard.");
}

void clearTrip(const char* reason) {
  motorTripped = false;
  criticalCount = 0;
  Serial.print("Trip cleared: ");
  Serial.println(reason);
}

void updateProtection() {
  if (currentStatus == "CRITICAL") {
    criticalCount++;
  } else {
    criticalCount = 0;
  }

  if (!motorTripped && criticalCount >= TRIP_AFTER) {
    tripMotor();
  }
}

// Like a motor starter: after a trip the knob has to go back to zero first
void checkLocalReset() {
  if (motorTripped && potPercent < 5) {
    clearTrip("speed knob returned to zero");
  }
}

void handleFogReply(const String& reply) {
  if (motorTripped && reply.indexOf("RESET_TRIP") >= 0) {
    clearTrip("restart command from fog dashboard");
  }
}

// ---------------- Fog communication ----------------

#if USE_GPS
// Adds ,"lat":..,"lon":..,"gps":0/1,"sats":n to the JSON being built
void appendLocation(String& json) {
  json += ",\"lat\":";
  json += String(gpsLat(SITE_LAT), 6);
  json += ",\"lon\":";
  json += String(gpsLon(SITE_LON), 6);
  json += ",\"gps\":";
  json += gpsHasFix() ? "1" : "0";
  json += ",\"sats\":";
  json += String(gpsSatellites());
}
#endif

String buildPayload(const String& status, bool withWindow) {
  String json;
  json.reserve(withWindow ? 2000 : 200);
  json += "{";
  json += "\"id\":\"" MACHINE_ID "\"";
  json += ",\"site\":\"" SITE_NAME "\"";
  json += ",\"vibration\":";
  json += String(vibration, 2);
  json += ",\"motorSpeed\":";
  json += String(motorSpeed);
  json += ",\"status\":\"";
  json += status;
  json += "\",\"motor\":\"";
  json += motorTripped ? "TRIPPED" : "RUNNING";
  json += "\"";

#if USE_GPS
  appendLocation(json);
#endif

  if (withWindow) {
    // window in milli-g, as integers to keep the message short
    json += ",\"w\":[";
    for (int i = 0; i < WINDOW; i++) {
      if (i > 0) json += ",";
      json += String((int)lroundf(windowData[i] * 1000.0));
    }
    json += "]";
  }

  json += "}";
  return json;
}

// Returns true if the fog server accepted the reading
bool sendTelemetry() {
  if (WiFi.status() != WL_CONNECTED) {
    fogOnline = false;
    return false;
  }

  HTTPClient http;
  http.begin(String(fogServer) + "/data");
  http.setConnectTimeout(HTTP_TIMEOUT_MS);
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.addHeader("Content-Type", "application/json");

  int httpCode = http.POST(buildPayload(currentStatus, true));

  if (httpCode == 200) {
    handleFogReply(http.getString());
    fogOnline = true;
  } else {
    Serial.print("Telemetry failed: ");
    Serial.println(httpCode > 0 ? String(httpCode) : http.errorToString(httpCode));
    fogOnline = false;
  }

  http.end();
  return fogOnline;
}

void storeReading() {
  if (backlogCount == BACKLOG_SIZE) {
    // Buffer full: drop the oldest reading
    backlogStart = (backlogStart + 1) % BACKLOG_SIZE;
    backlogCount--;
  }

  int index = (backlogStart + backlogCount) % BACKLOG_SIZE;
  backlog[index].vibration = vibration;
  backlog[index].pwm = motorSpeed;
  backlog[index].status = statusCode(currentStatus);
  backlog[index].takenAt = millis();
  backlogCount++;
}

// Uploads the oldest buffered readings; "age" lets the fog work out when each was taken
void flushBacklog() {
  int count = min(backlogCount, BATCH_SIZE);
  unsigned long now = millis();

  String json = "{\"id\":\"" MACHINE_ID "\",\"site\":\"" SITE_NAME "\"";
  json += ",\"readings\":[";

  for (int i = 0; i < count; i++) {
    Reading& r = backlog[(backlogStart + i) % BACKLOG_SIZE];
    if (i > 0) json += ",";
    json += "{\"v\":";
    json += String(r.vibration, 2);
    json += ",\"pwm\":";
    json += String(r.pwm);
    json += ",\"s\":\"";
    json += STATUS_NAMES[r.status];
    json += "\",\"age\":";
    json += String(now - r.takenAt);
    json += "}";
  }
  json += "]}";

  HTTPClient http;
  http.begin(String(fogServer) + "/data/batch");
  http.setConnectTimeout(HTTP_TIMEOUT_MS);
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.addHeader("Content-Type", "application/json");

  int httpCode = http.POST(json);

  if (httpCode == 200) {
    handleFogReply(http.getString());
    backlogStart = (backlogStart + count) % BACKLOG_SIZE;
    backlogCount -= count;
    Serial.print("Uploaded ");
    Serial.print(count);
    Serial.print(" buffered readings, ");
    Serial.print(backlogCount);
    Serial.println(" left");
  }

  http.end();
}

void sendCriticalAlert() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi not connected, critical alert not sent (local alarm is on).");
    return;
  }

  unsigned long now = millis();

  if (ALERT_COOLDOWN > 0 && lastAlertTime > 0 && now - lastAlertTime < ALERT_COOLDOWN) {
    Serial.println("Alert cooldown active.");
    return;
  }

  Serial.println();
  Serial.println("================================");
  Serial.println("SENDING CRITICAL ALERT TO FOG");
  Serial.println("================================");

  HTTPClient http;
  http.begin(String(fogServer) + "/alert");
  http.setConnectTimeout(HTTP_TIMEOUT_MS);
  http.setTimeout(8000);   // fog waits for the SMS to be handed over before replying
  http.addHeader("Content-Type", "application/json");

  String json = buildPayload("CRITICAL", false);
  Serial.print("Sending: ");
  Serial.println(json);

  int httpCode = http.POST(json);
  Serial.print("HTTP Response Code: ");
  Serial.println(httpCode);

  if (httpCode == 200) {
    Serial.print("Fog Response: ");
    Serial.println(http.getString());
    lastAlertTime = now;
    criticalSent = true;
  } else {
    Serial.println("Failed to contact Fog server, will retry next reading.");
  }

  http.end();
  Serial.println("================================");
}

// ---------------- LoRa uplink ----------------

#if USE_LORA
// Sends the reading as a 36-byte packet with band energies computed here
void sendLoraReading(bool newCritical) {
  UplinkPacket packet;
  packet.node = MACHINE_NUMBER;
  packet.levelMilliG = (uint16_t)min(65535L, lroundf(vibration * 1000.0f));
  packet.pwm = motorSpeed;
  packet.flags = statusCode(currentStatus);
  if (motorTripped) packet.flags |= FLAG_TRIPPED;
  if (newCritical) packet.flags |= FLAG_ALERT;

  float bands[BAND_COUNT];
  computeBandEnergies(windowData, WINDOW, 1000000.0f / WINDOW_STEP_US, bands);
  for (int b = 0; b < BAND_COUNT; b++) packet.bands[b] = (int16_t)lroundf(bands[b] * 1000.0f);

#if USE_GPS
  if (gpsHasFix()) packet.flags |= FLAG_GPS_FIX;
  packet.latE6 = (int32_t)lround(gpsLat(SITE_LAT) * 1e6);
  packet.lonE6 = (int32_t)lround(gpsLon(SITE_LON) * 1e6);
#else
  packet.latE6 = (int32_t)lround(SITE_LAT * 1e6);
  packet.lonE6 = (int32_t)lround(SITE_LON * 1e6);
#endif

  uint8_t command = loraExchange(packet, LORA_REPLY_WAIT_MS);
  fogOnline = true;   // on LoRa this means "last packet sent"
  if (command == CMD_RESET_TRIP && motorTripped) clearTrip("restart command over LoRa");

  Serial.print("LoRa packet sent, seq ");
  Serial.println(packet.seq);
}

void reportOverLora() {
  bool stateChanged = currentStatus != lastReportedStatus || motorTripped != lastReportedTrip;
  bool newCritical = currentStatus == "CRITICAL" && !criticalSent;

  readingsSinceReport++;
  if (!loraReady || (!stateChanged && readingsSinceReport < LORA_REPORT_EVERY)) return;

  sendLoraReading(newCritical);
  if (newCritical) criticalSent = true;
  readingsSinceReport = 0;
  lastReportedStatus = currentStatus;
  lastReportedTrip = motorTripped;
}
#endif

// ---------------- LCD ----------------

void updateLCD() {
  char line1[17];
  char line2[17];

  snprintf(line1, sizeof(line1), "V:%.2fg S:%3d%%", vibration, motorTripped ? 0 : speedPercent);

  const char* state = motorTripped ? "TRIPPED" : currentStatus.c_str();

#if USE_LORA
  snprintf(line2, sizeof(line2), "%-8s LoRa", state);
#else
  if (backlogCount > 0) {
    snprintf(line2, sizeof(line2), "%-8s B:%-4d", state, backlogCount);
  } else {
    snprintf(line2, sizeof(line2), "%-8s F:%s", state, fogOnline ? "OK" : "--");
  }
#endif

  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.print("                ");

  lcd.setCursor(0, 1);
  lcd.print(line2);
  lcd.print("                ");
}

// ---------------- Setup / loop ----------------

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println("       RESQFOG EDGE NODE");
  Serial.print("       ");
  Serial.print(MACHINE_ID);
  Serial.print(" - ");
  Serial.println(SITE_NAME);
  Serial.println("================================");

  Wire.begin(SDA_PIN, SCL_PIN);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("ResQFog ");
  lcd.print(MACHINE_ID);
  lcd.setCursor(0, 1);
  lcd.print("Starting...");

  writeRegister(0x6B, 0x00);   // wake up
  delay(100);
  writeRegister(0x1A, 0x01);   // DLPF 184 Hz, 1 kHz internal sample rate
  writeRegister(0x19, 0x00);   // sample rate divider 0
  writeRegister(0x1C, 0x00);   // accelerometer +-2 g
  Serial.println("MPU initialised.");

  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);
  pinMode(ENA_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(POT_PIN, INPUT);

  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW);
  analogWrite(ENA_PIN, 0);

#if USE_GPS
  gpsBegin();
  Serial.println("GPS started on UART2 (using installed location until a fix).");
#endif

  delay(1000);

#if USE_LORA
  loraReady = loraBegin();
  Serial.println(loraReady ? "LoRa radio ready." : "LoRa radio NOT found - check wiring.");
  lcd.clear();
  lcd.print(loraReady ? "LoRa ready" : "LoRa error");
  delay(1000);
#else
  connectWiFi();
  Serial.print("Fog server: ");
  Serial.println(fogServer);
#endif
  Serial.println("================================");

  lcd.clear();
}

void loop() {
#if USE_GPS
  gpsPoll();
#endif
#if !USE_LORA
  maintainWiFi();
#endif

  motorSpeed = map(analogRead(POT_PIN), 0, 4095, 0, 255);
  potPercent = map(motorSpeed, 0, 255, 0, 100);

  checkLocalReset();

  analogWrite(ENA_PIN, motorTripped ? 0 : motorSpeed);
  speedPercent = potPercent;

  if (millis() - lastSampleTime < SAMPLE_INTERVAL) return;
  lastSampleTime = millis();

  captureWindow();             // ~0.5 s; also leaves the latest reading in ax, ay, az
  vibration = calculateVibration();
  currentStatus = classifyVibration(vibration);

  updateProtection();

  Serial.print("Vibration: ");
  Serial.print(vibration, 2);
  Serial.print(" | Speed: ");
  Serial.print(motorTripped ? 0 : speedPercent);
  Serial.print("% | Status: ");
  Serial.print(currentStatus);
  if (motorTripped) Serial.print(" | MOTOR TRIPPED");
  Serial.println();

  // Local alarm first, before any network call
  if (currentStatus == "CRITICAL") {
    tone(BUZZER_PIN, 2000);
  } else if (motorTripped) {
    tone(BUZZER_PIN, 1500, 150);   // short beep every second while tripped
  } else {
    noTone(BUZZER_PIN);
  }

#if USE_LORA
  reportOverLora();
  if (currentStatus != "CRITICAL") criticalSent = false;
#else
  if (sendTelemetry()) {
    if (backlogCount > 0) flushBacklog();
  } else {
    storeReading();
  }

  if (currentStatus == "CRITICAL") {
    if (!criticalSent) sendCriticalAlert();
  } else {
    criticalSent = false;
  }
#endif

  updateLCD();
}
