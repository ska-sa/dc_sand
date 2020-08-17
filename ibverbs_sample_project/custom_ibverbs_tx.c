

#include <infiniband/verbs.h>
//#include <infiniband/verbs_exp.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <arpa/inet.h>// Used for getting the MAC address from an interface name
#include <sys/socket.h>// Used for getting the MAC address from an interface name
#include <ifaddrs.h> // Used for getting the MAC address from an interface name
#include <netpacket/packet.h>// Used for getting the MAC address from an interface name
#include <net/if_arp.h>// Used for getting the MAC address from an interface name
#include <net/ethernet.h>//MAC string to byte conversion

#define PORT_NUM 1
#define SQ_NUM_DESC 2048 /* maximum number of sends waiting for completion */
#define NUM_WE 4096
#define DESTINATION_IP_ADDRESS "10.100.18.7"
//Store MAC_ADDRESS as array of bytes as this is the easiest to convert to a network byte order
//#define DESTINATION_MAC_ADDRESS "1c:34:da:4b:93:92"
#define DESTINATION_MAC_ADDRESS {0x1c,0x34,0xda,0x4b,0x93,0x92}
#define SOURCE_IP_ADDRESS "10.100.18.9"
//Store MAC_ADDRESS - this should already be in network byte order
#define SOURCE_MAC_ADDRESS {0x1c,0x34,0xda,0x54,0x99,0xbc}
//#define SOURCE_MAC_ADDRESS "1c:34:da:54:99:bc"


/* template of packet to send - in this case icmp */
#define DST_MAC 0x00, 0x01, 0x02, 0x03, 0x04, 0x05
#define SRC_MAC 0xe4, 0x1d, 0x2d, 0xf3, 0xdd, 0xcc
#define ETH_TYPE 0x08, 0x00
#define IP_HDRS 0x45, 0x00, 0x00, 0x54, 0x00, 0x00, 0x40, 0x00, 0x40, 0x01, 0xaf, 0xb6
#define SRC_IP 0x0d, 0x07, 0x38, 0x66
#define DST_IP 0x0d, 0x07, 0x38, 0x7f
#define IP_OPT 0x08, 0x00, 0x59, 0xd0, 0x88
#define ICMP_HDR 0x2c, 0x00, 0x09, 0x52, 0xae, 0x96, 0x57, 0x00, 0x00

// char packet[] = {
//     DST_MAC , SRC_MAC, ETH_TYPE, IP_HDRS, SRC_IP, DST_IP, IP_OPT, ICMP_HDR,
//     0x00, 0x00, 0x62, 0x21, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
//     0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
//     0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35,
//     0x36, 0x37};

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
    uint8_t udp_datagram_payload[4096];
};

void populate_packet(struct network_packet * p_network_packet, struct ibv_context *p_context);


int main()
{
    struct ibv_device **dev_list;
    struct ibv_device *ib_dev;
    struct ibv_context *context;
    struct ibv_pd *pd;
    int ret;

    printf("Network Packet Size: %ld\n", sizeof(struct network_packet));

    /*1. Get the list of offload capable devices */
    dev_list = ibv_get_device_list(NULL);
    if (!dev_list)
    {
        printf("Failed to get devices list");
        exit(1);
    }
    
    /* In this example, we will use the first adapter (device) we find on the list (dev_list[0]) . You may change the code in case you have a setup with more than one adapter installed. */
    ib_dev = dev_list[1];
    
    printf("RDMA device[%d]: name=%s guid=%lld\n", 1, ibv_get_device_name(ib_dev), (unsigned long long)ntohl(ibv_get_device_guid(ib_dev)));
    if (!ib_dev)
    {
        printf("IB device not found\n");
        exit(1);
    }

    /* 2. Get the device context */
    /* Get context to device. The context is a descriptor and needed for resource tracking and operations */
    context = ibv_open_device(ib_dev);
    if (!context)
    {
        printf("Couldn't get context for %s\n", ibv_get_device_name(ib_dev));
        exit(1);
    }

    /* 3. Allocate Protection Domain */
    /* Allocate a protection domain to group memory regions (MR) and rings */
    pd = ibv_alloc_pd(context);
    if (!pd)
    {
        printf("Couldn't allocate PD\n");
        exit(1);
    }

    /* 4. Create Complition Queue (CQ) */
    struct ibv_cq *cq;
    cq = ibv_create_cq(context, SQ_NUM_DESC, NULL, NULL, 0);
    if (!cq)
    {
        printf("Couldn't create CQ %d\n", errno);
        exit (1);
    }

    /* 5. Initialize QP */
    struct ibv_qp *qp;
    struct ibv_qp_init_attr qp_init_attr = {
        .qp_context = NULL,
        /* report send completion to cq */
        .send_cq = cq,
        .recv_cq = cq,
        .cap = {
            /* number of allowed outstanding sends without waiting for a completion */
            .max_send_wr = SQ_NUM_DESC,
            /* maximum number of pointers in each descriptor */
            .max_send_sge = 1,
            /* if inline maximum of payload data in the descriptors themselves */
            .max_inline_data = 512,
            .max_recv_wr = 0
        },
        .qp_type = IBV_QPT_RAW_PACKET,
    };

    /* 6. Create Queue Pair (QP) - Send Ring */
    qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp)
    {
        printf("Couldn't create RSS QP\n");
        exit(1);
    }

    /* 7. Initialize the QP (receive ring) and assign a port */
    struct ibv_qp_attr qp_attr;
    int qp_flags;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_flags = IBV_QP_STATE | IBV_QP_PORT;
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.port_num = 1;
    ret = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (ret < 0)
    {
        printf("failed modify qp to init\n");
        exit(1);
    }
    memset(&qp_attr, 0, sizeof(qp_attr));

    /* 8. Move the ring to ready to send in two steps (a,b) */
    /* a. Move ring state to ready to receive, this is needed to be able to move ring to ready to send even if receive queue is not enabled */
    qp_flags = IBV_QP_STATE;
    qp_attr.qp_state = IBV_QPS_RTR;
    ret = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (ret < 0)
    {
        printf("failed modify qp to receive\n");
        exit(1);
    }

    /* b. Move the ring to ready to send */
    qp_flags = IBV_QP_STATE;
    qp_attr.qp_state = IBV_QPS_RTS;
    ret = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (ret < 0)
    {
        printf("failed modify qp to receive\n");
        exit(1);
    }

    /* 9. Allocate Memory */
    int buf_size = sizeof(struct network_packet)*SQ_NUM_DESC; /* maximum size of data to be access directly by hw */
    void *buf;
    buf = malloc(buf_size);
    if (!buf)
    {
        printf("Could not allocate memory\n");
        exit(1);   
    }

    /* 10. Register the user memory so it can be accessed by the HW directly */
    struct ibv_mr *mr;
    mr = ibv_reg_mr(pd, buf, buf_size, IBV_ACCESS_LOCAL_WRITE);
    if (!mr)
    {
        printf("Couldn't register mr\n");
        exit(1);
    }

    struct network_packet packet;
    memset(&packet,0x00,sizeof(struct network_packet));
    populate_packet(&packet, context);
    memcpy(buf, &packet, sizeof(struct network_packet));
    
    int n;
    struct ibv_sge sg_entry;
    struct ibv_send_wr wr[NUM_WE], *bad_wr;
    int msgs_completed;
    struct ibv_wc wc;

    /* scatter/gather entry describes location and size of data to send*/
    sg_entry.addr = (uint64_t)buf;
    sg_entry.length = sizeof(struct network_packet);
    sg_entry.lkey = mr->lkey;
    memset(wr, 0, sizeof(wr[0])*NUM_WE);

    /*
     * descriptor for send transaction - details:
     * - how many pointer to data to use
     * - if this is a single descriptor or a list (next == NULL single)
     * - if we want inline and/or completion
     */

    for (size_t i = 0; i < NUM_WE - 1; i++)
    {
        wr[i].num_sge = 1;
        wr[i].sg_list = &sg_entry;
        wr[i].next = &wr[i+1];
        wr[i].opcode = IBV_WR_SEND;
    }

    wr[NUM_WE-1].num_sge = 1;
    wr[NUM_WE-1].sg_list = &sg_entry;
    wr[NUM_WE-1].next = NULL;
    wr[NUM_WE-1].opcode = IBV_WR_SEND;


    /* 10. Send Operation */
    while(1) 
    {
        /*
         * inline means data will be copied to space pre-allocated in descriptor
         * as long as it is small enough. otherwise pointer reference will be used.
         * see max_inline_data = 512 above.
         */
        //wr.send_flags = IBV_SEND_INLINE;
    
        /*
        * we ask for a completion every half queue. only interested in completions to monitor progress.
        */
        if ( (n % (SQ_NUM_DESC/2)) == 0) {
            wr[0].wr_id = n;
            wr[0].send_flags |= IBV_SEND_SIGNALED;
        }

        /* push descriptor to hardware */

        ret = ibv_post_send(qp, &wr[0], &bad_wr);
        if (ret < 0) 
        {
            printf("failed in post send\n");
            exit(1);
        }
        n+=NUM_WE;

        /* poll for completion after half ring is posted */
        if ( (n % (SQ_NUM_DESC/2)) == 0 && n > 0)
        {
            msgs_completed = ibv_poll_cq(cq, 1, &wc);
            if (msgs_completed > 0) 
            {
                //printf("completed message %ld\n", wc.wr_id);
            }
            else if (msgs_completed < 0) 
            {
                printf("Polling error\n");
                exit(1);
            }
        }

    }

    printf("We are done\n");
    return 0;
}

void populate_packet(struct network_packet * p_network_packet, struct ibv_context * p_context){
    

    /* 1. Determine Interface IP and MAC Address - need to determine this from the context*/

    //struct ifaddrs *ifap, *ifa;
    //struct sockaddr_in *sa;
    //char *addr;
    //union ibv_gid gid;

    //Not sure why but the second argument must be 2
    //int rc = ibv_query_gid(p_context, 1, 2, &gid);
    //if (rc) {
	//    printf("Error, failed to query GID index %d of port %d in device '%s'\n",1, 0, ibv_get_device_name(p_context->device));
    //}
    //printf("GID: GID Prefix: %d %d %d %d %d %d %d %d\n",(uint32_t)gid.raw[0], (uint32_t)gid.raw[1], (uint32_t)gid.raw[2], (uint32_t)gid.raw[3], (uint32_t)gid.raw[4], (uint32_t)gid.raw[5], (uint32_t)gid.raw[6], (uint32_t)gid.raw[7]);
    //printf("GID: Subnet Prefix: %d %d %d %d %d %d %d %d\n", (uint32_t)gid.raw[8], (uint32_t)gid.raw[9], (uint32_t)gid.raw[10], (uint32_t)gid.raw[11], (uint32_t)gid.raw[12], (uint32_t)gid.raw[13], (uint32_t)gid.raw[14] , (uint32_t)gid.raw[15]);

    //IP Layer

    //This value is hardcoded - have not bothered to look into it
    p_network_packet->ip_packet_version_and_ihl = 0x45;
    
    //These values allow for differentiating type of service and congestion level - I think this is not used in the MeerKAT network, so it is just left at 0
    p_network_packet->ip_packet_dscp_and_ecn = 0x0;

    //This is the packet length - it includes the IP header and data while excluding the ethernet frame fields.
    uint16_t u16IPPacketLengthBytes = sizeof(*p_network_packet) 
            - sizeof(p_network_packet->ethernet_frame_dest_mac) 
            - sizeof(p_network_packet->ethernet_frame_src_mac)
            - sizeof(p_network_packet->ethernet_frame_ether_type);
    p_network_packet->ip_packet_total_length = htons(u16IPPacketLengthBytes);

    //If an IP packet is fragmented during transmission, this field contains a unique number identifying the original packet when packets are reassembled.
    p_network_packet->ip_packet_identification = 0;

    //This specifies if a packet can be fragmented and what the offset of the current fragment is. We set a flag to disable fragmentation
    p_network_packet->ip_packet_flags_and_fragment_offset = htons(0x4000);

    //TTL is well explained - google it. Set to eight for now, but that was more a guess. May need to be reworked in the future
    p_network_packet->ip_packet_ttl = 8;

    //17 represents the UDP protocol
    p_network_packet->ip_packet_protocol = 17;

    //Set IP addresses
    p_network_packet->ip_packet_dest_address = inet_addr(DESTINATION_IP_ADDRESS);
    p_network_packet->ip_packet_src_address = inet_addr(SOURCE_IP_ADDRESS);

    //Calculating the checksum - break the header into 16 bit chunks. Sum these 16 bit chunks together and then 1 compliment them.
    p_network_packet->ip_packet_checksum = 0; //Must start off as zero for the calculation
    
    //1. Array of 16 bit chunks
    uint16_t * pu16IPHeader = (uint16_t *) &p_network_packet->ip_packet_version_and_ihl;

    //2.1 16-bit sum of data - we store it as a 32 bit number as the carry values need to be used.
    uint32_t u32Sum = 0;
    for (size_t i = 0; i < 10; i++)
    {
        u32Sum += ntohs(pu16IPHeader[i]);//Remember network byte order
    }
    //2.2 Compensate for carry - every time a carry occurs, add one to the u32Sum. Can do this at the end as follows:
    //This has not actually been tested yet
    //At first glance this could be an if statement, however if adding all the carry bits causes an additional carry, then this step needs to occur again, this is why a while loop is necessary. 
    while (u32Sum > 0xffff){
        u32Sum = (u32Sum & 0xffff) + (u32Sum >> 16);
    }

    //3. 1s compliment the data
    uint16_t u16SumComplimented = ~(uint16_t)u32Sum;

    //4. Store checksum in packet.
    p_network_packet->ip_packet_checksum = htons(u16SumComplimented);

    //Ethernet Layer
    uint8_t pu8DestMacAddress[6] = DESTINATION_MAC_ADDRESS;
    uint8_t pu8SrcMacAddress[6] = SOURCE_MAC_ADDRESS;
    for (size_t i = 0; i < 6; i++)
    {   
        p_network_packet->ethernet_frame_dest_mac[i] = pu8DestMacAddress[i];
        p_network_packet->ethernet_frame_src_mac[i] = pu8SrcMacAddress[i];
    }
    p_network_packet->ethernet_frame_ether_type = htons(0x0800);
    
    uint8_t * p_temp = (uint8_t*) & p_network_packet->ip_packet_src_address;
    printf("Source IP: %d.%d.%d.%d\n",(int32_t)p_temp[0],(int32_t)p_temp[1],(int32_t)p_temp[2],(int32_t)p_temp[3]);

    p_temp = (uint8_t*) & p_network_packet->ip_packet_dest_address;
    printf("Destination IP: %d.%d.%d.%d\n",(int32_t)p_temp[0],(int32_t)p_temp[1],(int32_t)p_temp[2],(int32_t)p_temp[3]);
    
    p_temp = (uint8_t*) & p_network_packet->ethernet_frame_dest_mac;
    printf("Destination MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",(int32_t)p_temp[0],(int32_t)p_temp[1],(int32_t)p_temp[2],(int32_t)p_temp[3],(int32_t)p_temp[4],(int32_t)p_temp[5]);
    
    p_temp = (uint8_t*) & p_network_packet->ethernet_frame_src_mac;
    printf("Source MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",(int32_t)p_temp[0],(int32_t)p_temp[1],(int32_t)p_temp[2],(int32_t)p_temp[3],(int32_t)p_temp[4],(int32_t)p_temp[5]);
    
    printf("Packet Length excluding frame: %d bytes\n",(int32_t)u16IPPacketLengthBytes);
    //Ethernet Layer
    //p_network_packet->ethernet_frame_dest_mac = 
    //p_network_packet->ethernet_frame_src_mac = 

    // getifaddrs (&ifap);
    // for (ifa = ifap; ifa; ifa = ifa->ifa_next) {
    //     if (ifa->ifa_addr && ifa->ifa_addr->sa_family==AF_INET) {
    //         sa = (struct sockaddr_in *) ifa->ifa_addr;
    //         addr = inet_ntoa(sa->sin_addr);
    //         uint8_t addr_octets[4];
    //         ((uint32_t *) addr_octets)[0] = *(uint32_t*) &sa->sin_addr;
            
    //         if(addr_octets[0] == gid.raw[12] && addr_octets[1] == gid.raw[13] && addr_octets[2] == gid.raw[14] && addr_octets[3] == gid.raw[15])
    //         {
    //             printf("Interface: %s\tAddress: %s\n", ifa->ifa_name, addr);
    //             struct sockaddr_ll *ll = (struct sockaddr_ll *) ifa->ifa_addr;
    //             if (ll->sll_hatype == ARPHRD_ETHER && ll->sll_halen == 6)
    //             {
    //                 memcpy(&p_network_packet->ethernet_frame_src_mac, ll->sll_addr, 6);
    //             }
    //             break;
    //         }
    //     }
    // }

    // p_network_packet->ip_packet_src_address = inet_addr(addr);
    // printf("src ip address: %d\n", p_network_packet->ip_packet_src_address);
    // printf("src mac address: %d:%d:%d:%d:%d:%d\n",
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[0],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[1],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[2],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[3],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[4],
    //         (uint32_t)p_network_packet->ethernet_frame_src_mac[5]);

    // freeifaddrs(ifap);
}
