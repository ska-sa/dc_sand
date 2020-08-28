#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include <stdint.h>
#include <infiniband/verbs.h>

/* Functions in this library are used for converting IP address strings to character arrays and for converting data 
 * between network and host byte order.
 */
#include <arpa/inet.h> 

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

struct ibv_device * get_ibv_device_from_ip(uint8_t * u8PortNumber, char * strLocalIpAddress);

struct ibv_device * get_ibv_device_from_ip(uint8_t * u8PortIndex, char * strLocalIpAddress){
    
    struct ibv_device **dev_list;
    struct ibv_device *ib_dev;
    struct ibv_context *p_context;
    int iNumDevices;
    uint8_t u8DeviceFound = 0;

    /* 1. Store the source IP address as four octets for ease of comparison */
    uint8_t pu8SourceAddrOctets[4];
    ((uint32_t *) pu8SourceAddrOctets)[0] = inet_addr(strLocalIpAddress);

    /* 2. Get the list of offload capable devices */
    dev_list = ibv_get_device_list(&iNumDevices);
    if (!dev_list)
    {
        printf("Failed to get devices list");
        exit(1);
    }
    
    /* 3. Iterate through all offload capable devices to find the one with the correct IP address*/
    for (size_t i = 0; i < iNumDevices; i++)
    {   
        ib_dev = dev_list[i];
        printf("RDMA device[%ld]: name=%s\n", i, ibv_get_device_name(ib_dev));
        if (!ib_dev)
        {
            printf("IB device not found\n");
            exit(1);
        }
        
        /* 3.1 The device context is required by the ibv_query_gid() function */
        p_context = ibv_open_device(ib_dev);
        if (!p_context)
        {
            printf("Couldn't get context for %s\n", ibv_get_device_name(ib_dev));
            exit(1);
        }

        /* 3.2 Iterate through all the ports of each device*/
        union ibv_gid gid;
        *u8PortIndex = 1;
        while(1)
        {
            /* 3.2.1 Get the port GID*/
            //Not sure why but the second argument works when set to 2 for my test configuration, but 1 for others. If your IP address is not detected, try change the value to 1.
            int rc = ibv_query_gid(p_context, *u8PortIndex, 2, &gid);
            if (rc) 
            {
                break;
            }
            printf("\tPhysical Port: %d\n",*u8PortIndex);
            printf("\t\tGID: GID Prefix: %d %d %d %d %d %d %d %d\n",(uint32_t)gid.raw[0], (uint32_t)gid.raw[1], (uint32_t)gid.raw[2], (uint32_t)gid.raw[3], (uint32_t)gid.raw[4], (uint32_t)gid.raw[5], (uint32_t)gid.raw[6], (uint32_t)gid.raw[7]);
            printf("\t\tGID: Subnet Prefix: %d %d %d %d %d %d %d %d\n", (uint32_t)gid.raw[8], (uint32_t)gid.raw[9], (uint32_t)gid.raw[10], (uint32_t)gid.raw[11], (uint32_t)gid.raw[12], (uint32_t)gid.raw[13], (uint32_t)gid.raw[14] , (uint32_t)gid.raw[15]);
            printf("\t\tIP Address From GID: %d.%d.%d.%d\n",(uint32_t)gid.raw[12], (uint32_t)gid.raw[13], (uint32_t)gid.raw[14] , (uint32_t)gid.raw[15]);

            /* 3.2.2 Compare the fields in the GID that correspond to IP address with the expected IP address */
            if(pu8SourceAddrOctets[0] == gid.raw[12] && pu8SourceAddrOctets[1] == gid.raw[13] && pu8SourceAddrOctets[2] == gid.raw[14] && pu8SourceAddrOctets[3] == gid.raw[15])
            {
                u8DeviceFound = 1;
                break;
            }

            *u8PortIndex = *u8PortIndex + 1;
        }

        /* 3.3 Cleanup */
        ibv_close_device(p_context);
        if(u8DeviceFound){
            break;
        }

    }

    /* 4. Set pointer to NULL if no device is found for a safe exit condition */
    if(u8DeviceFound == 0){
        *u8PortIndex = 0;
        ib_dev = NULL;
    }

    /* 5. Cleanup */
    ibv_free_device_list(dev_list);

    /* 6. Return device pointer*/
    return ib_dev;
}

#endif