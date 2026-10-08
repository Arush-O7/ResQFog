// LoRa packet shared by the edge node and the gateway.
// Keep this file identical in edge/resqfog_edge and edge/resqfog_lora_gateway.

#pragma once
#include <stdint.h>

#define LORA_PACKET_VERSION 1

// Uplink: edge node -> gateway, 36 bytes
struct __attribute__((packed)) UplinkPacket {
  uint8_t version;      // LORA_PACKET_VERSION
  uint8_t node;         // pump number, PUMP-01 -> 1
  uint16_t seq;         // increases by one per packet
  uint16_t levelMilliG; // vibration level v in milli-g
  uint8_t pwm;          // motor PWM 0-255
  uint8_t flags;        // bits 0-1 state (0 normal, 1 warning, 2 critical)
                        // bit 2 motor tripped, bit 3 GPS fix, bit 4 new critical (alert)
  int16_t bands[10];    // log10 band energies x 1000
  int32_t latE6;        // latitude x 1e6 (0 if no location)
  int32_t lonE6;        // longitude x 1e6
};

// Downlink: gateway -> edge node, sent right after an uplink
struct __attribute__((packed)) DownlinkPacket {
  uint8_t version;
  uint8_t node;
  uint8_t command;      // 1 = restart a tripped motor
};

#define FLAG_TRIPPED   0x04
#define FLAG_GPS_FIX   0x08
#define FLAG_ALERT     0x10
#define CMD_RESET_TRIP 1

// SX1278 (Ra-02) wiring on the ESP32 VSPI bus
#define LORA_SCK   18
#define LORA_MISO  19
#define LORA_MOSI  23
#define LORA_NSS   5
#define LORA_RST   14
#define LORA_DIO0  2

// Use the frequency your module and local regulations allow
// (433 MHz for SX1278 / Ra-02 modules).
#define LORA_FREQ  433E6
#define LORA_SF    7
#define LORA_BW    125E3
#define LORA_SYNC  0x34
