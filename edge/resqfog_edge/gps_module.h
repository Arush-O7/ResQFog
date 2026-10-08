// Optional NEO-6M GPS (enable with USE_GPS 1 in resqfog_edge.ino).
// Wiring: GPS TX -> GPIO 16, GPS RX -> GPIO 17, VCC 3.3-5 V, GND.
// Library: TinyGPSPlus (Library Manager)
//
// Without a fix (indoors, or before the first fix) the node reports the
// coordinates entered at installation and marks them as "fixed".

#pragma once
#include <TinyGPSPlus.h>

#define GPS_RX_PIN 16
#define GPS_TX_PIN 17
#define GPS_BAUD 9600

static HardwareSerial gpsSerial(2);
static TinyGPSPlus gps;

static void gpsBegin() {
  gpsSerial.setRxBufferSize(1024);
  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
}

// Call often; the HTTP and window code block for a while each second
static void gpsPoll() {
  while (gpsSerial.available()) gps.encode(gpsSerial.read());
}

static bool gpsHasFix() {
  return gps.location.isValid() && gps.location.age() < 5000;
}

static double gpsLat(double fallback) { return gpsHasFix() ? gps.location.lat() : fallback; }
static double gpsLon(double fallback) { return gpsHasFix() ? gps.location.lng() : fallback; }
static int gpsSatellites() { return gps.satellites.isValid() ? gps.satellites.value() : 0; }
