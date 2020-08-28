#ifndef NETWORK_PACKET_H
#define NETWORK_PACKET_H

#include <stdint.h>

#define PAYLOAD_SIZE_BYTES 4096

struct __attribute__((__packed__)) network_packet {
    uint8_t ethernet_frame_dest_mac[6];
    uint8_t ethernet_frame_src_mac[6];
    uint16_t ethernet_frame_ether_type;
    
    uint8_t ip_packet_version_and_ihl;
    uint8_t ip_packet_dscp_and_ecn;
    uint16_t ip_packet_total_length;
    uint16_t ip_packet_identification;
    uint16_t ip_packet_flags_and_fragment_offset;
    uint8_t ip_packet_ttl;
    uint8_t ip_packet_protocol;
    uint16_t ip_packet_checksum;
    uint32_t ip_packet_src_address;
    uint32_t ip_packet_dest_address;

    uint16_t upd_datagram_src_port;
    uint16_t upd_datagram_dest_port;
    uint16_t upd_datagram_length;
    uint16_t upd_datagram_checksum;
    uint8_t udp_datagram_payload[PAYLOAD_SIZE_BYTES];
};

#endif