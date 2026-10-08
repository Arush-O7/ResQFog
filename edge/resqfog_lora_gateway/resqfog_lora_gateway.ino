// ResQFog LoRa gateway - ESP32 + SX1278 next to the fog server
//
// Receives 36-byte packets from edge nodes that run with USE_LORA 1,
// forwards each one to the fog server as the same JSON a Wi-Fi node sends
// (band energies in "f" instead of the raw window), and sends restart
// commands back to a node right after its next packet.
//
// Libraries: LoRa by Sandeep Mistry (Library Manager)
// Copy secrets.example.h to secrets.h and set Wi-Fi and FOG_SERVER.

#include <WiFi.h>
#include <HTTPClient.h>
#include <SPI.h>
#include <LoRa.h>
#include "lora_packet.h"
#include "secrets.h"

const char* STATUS_NAMES[] = {"NORMAL", "WARNING", "CRITICAL"};

uint8_t pendingCommand[256] = {0};   // per node number

unsigned long received = 0;
unsigned long forwarded = 0;

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.print(" connected, IP ");
  Serial.println(WiFi.localIP());
}

bool loraBegin() {
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_NSS);
  LoRa.setPins(LORA_NSS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(LORA_FREQ)) return false;
  LoRa.setSpreadingFactor(LORA_SF);
  LoRa.setSignalBandwidth(LORA_BW);
  LoRa.setCodingRate4(5);
  LoRa.setSyncWord(LORA_SYNC);
  LoRa.enableCrc();
  return true;
}

void sendDownlink(uint8_t node, uint8_t command) {
  DownlinkPacket reply = {LORA_PACKET_VERSION, node, command};
  LoRa.beginPacket();
  LoRa.write((const uint8_t*)&reply, sizeof(reply));
  LoRa.endPacket();
}

String toJson(const UplinkPacket& p, int rssi, float snr, bool withFeatures) {
  char id[12];
  snprintf(id, sizeof(id), "PUMP-%02u", p.node);
  uint8_t state = p.flags & 0x03;
  if (state > 2) state = 0;

  String json = "{\"id\":\"";
  json += id;
  json += "\",\"vibration\":";
  json += String(p.levelMilliG / 1000.0, 3);
  json += ",\"motorSpeed\":";
  json += String(p.pwm);
  json += ",\"status\":\"";
  json += STATUS_NAMES[state];
  json += "\",\"motor\":\"";
  json += (p.flags & FLAG_TRIPPED) ? "TRIPPED" : "RUNNING";
  json += "\"";

  if (p.latE6 != 0 || p.lonE6 != 0) {
    json += ",\"lat\":";
    json += String(p.latE6 / 1e6, 6);
    json += ",\"lon\":";
    json += String(p.lonE6 / 1e6, 6);
    json += ",\"gps\":";
    json += (p.flags & FLAG_GPS_FIX) ? "1" : "0";
  }

  json += ",\"via\":\"lora\",\"rssi\":";
  json += String(rssi);
  json += ",\"snr\":";
  json += String(snr, 1);
  json += ",\"seq\":";
  json += String(p.seq);

  if (withFeatures) {
    json += ",\"f\":[";
    for (int b = 0; b < 10; b++) {
      if (b > 0) json += ",";
      json += String(p.bands[b] / 1000.0, 3);
    }
    json += "]";
  }
  json += "}";
  return json;
}

String post(const char* path, const String& json) {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  HTTPClient http;
  http.begin(String(FOG_SERVER) + path);
  http.setConnectTimeout(1500);
  http.setTimeout(8000);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(json);
  String reply = code == 200 ? http.getString() : "";
  http.end();
  return reply;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("ResQFog LoRa gateway");

  connectWiFi();

  if (!loraBegin()) {
    Serial.println("LoRa radio not found - check wiring");
    while (true) delay(1000);
  }
  Serial.println("Listening for edge nodes");
}

void loop() {
  int size = LoRa.parsePacket();
  if (size != sizeof(UplinkPacket)) return;

  UplinkPacket packet;
  LoRa.readBytes((uint8_t*)&packet, sizeof(packet));
  if (packet.version != LORA_PACKET_VERSION) return;

  int rssi = LoRa.packetRssi();
  float snr = LoRa.packetSnr();
  received++;

  // The node listens for 300 ms after sending, so reply before anything slow
  if (pendingCommand[packet.node]) {
    sendDownlink(packet.node, pendingCommand[packet.node]);
    Serial.printf("Downlink to node %u: command %u\n", packet.node, pendingCommand[packet.node]);
    pendingCommand[packet.node] = 0;
  }

  String reply = post("/data", toJson(packet, rssi, snr, true));
  if (reply.length()) forwarded++;
  if (reply.indexOf("RESET_TRIP") >= 0) pendingCommand[packet.node] = CMD_RESET_TRIP;

  if (packet.flags & FLAG_ALERT) post("/alert", toJson(packet, rssi, snr, false));

  Serial.printf("Node %u seq %u: %.2f g, RSSI %d dBm, SNR %.1f dB (%lu received, %lu forwarded)\n",
                packet.node, packet.seq, packet.levelMilliG / 1000.0, rssi, snr, received, forwarded);
}
