// Optional LoRa link (enable with USE_LORA 1 in resqfog_edge.ino).
// The node sends a 36-byte packet to the gateway (edge/resqfog_lora_gateway)
// and listens briefly for a reply that may carry a restart command.
// Library: LoRa by Sandeep Mistry (Library Manager)

#pragma once
#include <SPI.h>
#include <LoRa.h>
#include "lora_packet.h"

static uint16_t loraSeq = 0;

static bool loraBegin() {
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

// Sends one uplink, then waits up to waitMs for a downlink.
// Returns the command from the gateway, or 0 if there was none.
static uint8_t loraExchange(UplinkPacket& packet, uint16_t waitMs) {
  packet.version = LORA_PACKET_VERSION;
  packet.seq = loraSeq++;

  LoRa.beginPacket();
  LoRa.write((const uint8_t*)&packet, sizeof(packet));
  LoRa.endPacket();

  unsigned long start = millis();
  while (millis() - start < waitMs) {
    if (LoRa.parsePacket() == sizeof(DownlinkPacket)) {
      DownlinkPacket reply;
      LoRa.readBytes((uint8_t*)&reply, sizeof(reply));
      if (reply.version == LORA_PACKET_VERSION && reply.node == packet.node) {
        return reply.command;
      }
    }
  }
  return 0;
}
