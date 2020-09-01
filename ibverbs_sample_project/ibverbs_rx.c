/**
 * @file    ibverbs_rx.c
 *
 * @brief   Sample program that demonstrates how to receive raw UDP ethernet network data at high data rates using the 
 *          ibverbs library.
 * 
 * This program uses the ibverbs API to receive raw UDP ethernet data from a Mellanox NIC. This implementation both 
 * bypasses the kernel and is CPU light. Once received, a packet is DMAd directly from the NIC to a userpace buffer 
 * without CPU intervention. With this approach, very high receive speeds can be achieved (speeds >90 Gbps have been 
 * observed). 
 * 
 * The ibverbs library requires steering rules to be defined to determine which packets the NIC must intercept in this 
 * manner and which must be passed on to the kernel. This program demonstrates how to do this.
 * 
 * When examining this example, the ibverbs_tx.c file should be read before the ibverbs_rx.c file as it was created
 * first and explains some of the concepts in more detail.
 * 
 * @author  Gareth Callanan
 *          South African Radio Astronomy Observatory(SARAO)
 */

#include <infiniband/verbs.h>
/* Functions in this library are used for converting IP address strings to character arrays and for converting data 
 * between network and host byte order.
 */
#include <arpa/inet.h> 
#include <sys/time.h>   //For timing functions

#include "common_functions.h"

#define RQ_NUM_DESC 2048
#define ENTRY_SIZE 9000 /* The maximum size of each received packet - set to jumbo frame */
#define SOURCE_IP_ADDRESS "10.100.18.9"
#define DESTINATION_IP_ADDRESS "10.100.18.7"
#define UDP_PORT 7708

int main()
{
    struct ibv_device *ib_dev;
    struct ibv_context *context;
    struct ibv_pd *pd;
    uint8_t u8PortNum;

    int iReturnValue;

    /* 1. Get correct device and physical port number from source IP address specified by SOURCE_IP_ADDRESS. */
    ib_dev = get_ibv_device_from_ip(&u8PortNum, SOURCE_IP_ADDRESS);
    if(ib_dev == NULL){
        printf("No NIC with matching SOURCE_IP_ADDRESS found\n");
        exit(1);
    }

    /* 2. Get the device context. Same as in ibverbs_tx.c. */
    context = ibv_open_device(ib_dev);
    if (!context) {
        printf("Couldn't get context for %s\n",ibv_get_device_name(ib_dev));
        exit(1);
    }

    /* 3. Allocate Protection Domain. Same as in ibverbs_tx.c
     */
    pd = ibv_alloc_pd(context);
    if (!pd) {
        printf("Couldn't allocate PD\n");
        exit(1);
    }

    /* 4. Create Completion Queue (CQ). Similar to the one in ibverbs_tx.c. The ibverbs_tx.c CQ receives a completion
     * upon succesful transmission of a packet. This CQ receives a completion once a packet has been received and copied
     * into the specified MR buffer.
     */
    struct ibv_cq *cq_recv;
    cq_recv = ibv_create_cq(context, RQ_NUM_DESC, NULL, NULL, 0);

    if (!cq_recv) 
    {
        printf("Couldn't create cq_recv\n");
        exit (1);
    }

    /* 5.1 Initialize queue pair structs. Similar to the one in ibverbs_tx.c.
     */ 
    struct ibv_qp *qp;
    struct ibv_qp_init_attr qp_init_attr = {
        .qp_context = NULL,
        /* report receive completion to cq */
        .send_cq = cq_recv,
        .recv_cq = cq_recv,
        .cap = {
            /* no send ring */
            .max_send_wr = 0,
            /* maximum number of packets in ring */
            .max_recv_wr = RQ_NUM_DESC,
            /* only one pointer per descriptor */
            .max_recv_sge = 1,
        },
        .qp_type = IBV_QPT_RAW_PACKET,
    };

    /* 5.2 Create Queue Pair (QP). Similar to the one in ibverbs_tx.c.*/
    qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) 
    {
        printf("Couldn't create RSS QP\n");
        exit(1);
    }

    /* 5.3 Initialize the QP and assign the correct physical port. Similar to the one in ibverbs_tx.c. */
    struct ibv_qp_attr qp_attr;
    int qp_flags;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_flags = IBV_QP_STATE | IBV_QP_PORT;
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.port_num = u8PortNum; /*I have never had this value equal to anything other than 1, I have a niggling \
    concern that if it equals another number things will not work;*/
    iReturnValue = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (iReturnValue) 
    {
        printf("failed modify qp to init %d %d\n",iReturnValue, errno);
        exit(1);
    }
    memset(&qp_attr, 0, sizeof(qp_attr));

    /* 6. Move this ring to a "ready" state. Only the ready to receive state need to be entered. */
    /* 6.1. Move ring state to ready to receive */
    qp_flags = IBV_QP_STATE;
    qp_attr.qp_state = IBV_QPS_RTR;
    iReturnValue = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (iReturnValue) 
    {
        printf("failed modify qp to receive %d %d\n",iReturnValue,errno);
        exit(1);
    }

    /* 7. Allocate memory buffer - This is user space memory that will have packets written to it by the NIC */
    int iDataBufferSize = ENTRY_SIZE*RQ_NUM_DESC; /* maximum size of data to be accessed by hardware */
    void *pDataBuffer;
    pDataBuffer = malloc(iDataBufferSize);
    if (!pDataBuffer) 
    {
        printf("Couldn't allocate memory\n");
        exit(1);
    }

    /* 8. Register the user memory so it can be accessed by the NIC directly. Similar to the one in ibverbs_tx.c.
     */
    struct ibv_mr *mr;
    mr = ibv_reg_mr(pd, pDataBuffer, iDataBufferSize, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) 
    {
        printf("Couldn't register mr\n");
        exit(1);
    }

    /* 9. Register steering rule. A steering rule can be thought of as a filter for the NIC. The steering rule
     * specifies properties within the ethernet/idp/udp headers that the NIC can filter on. It can filter on part or the
     * entirety of a field depending on how a bitmask is set. If the NIC receives a packet that matches the steering 
     * rule, it redirects that packet away from the kernel network stack to a user space buffer associated with a
     * designated QP instead.  
     * 
     * Below, a steering rule has been created that filters packets with a specific source and destination ip address
     * as well as a specific destination port number.
     */
    
    /*
     * 9.1 The first step is creating a flow rule struct. This struct consists of multiple ibv structs that define the 
     * various headers that will be used to create a steering rule. In this case, the IP header(for IP addresses) and
     * UDP header(for port numbers) are going to be filtered on and thus the corresponding ibv_flow "sub-structs" are
     * part of the flow_rule struct.
     */
    struct
    {
        struct ibv_flow_attr attr;
        struct ibv_flow_spec_eth eth __attribute__((packed));/* At one stage the flow rule would not work without this 
        struct, I am not sure if this has been fixed, but I have left it here for safety even though it is not strictly
        required */
        struct ibv_flow_spec_ipv4 ip __attribute__((packed));
        struct ibv_flow_spec_tcp_udp udp __attribute__((packed));
    } __attribute__((packed)) flow_rule;
    memset(&flow_rule, 0, sizeof(flow_rule));

    // 9.1.1 General Rule Set Up
    flow_rule.attr.type = IBV_FLOW_ATTR_NORMAL;
    flow_rule.attr.priority = 0;
    flow_rule.attr.size = sizeof(flow_rule);
    flow_rule.attr.num_of_specs = 3;
    flow_rule.attr.port = 1;

    // 9.1.2 Set L2 Layer flow rules
    flow_rule.eth.type = IBV_FLOW_SPEC_ETH;
    flow_rule.eth.size = sizeof(flow_rule.eth);

    // 9.1.3 Set L3 IP Layer flow rules. The mask is set to non-zero for fields we want to filter on.
    flow_rule.ip.type = IBV_FLOW_SPEC_IPV4;
    flow_rule.ip.size = sizeof(flow_rule.ip);
    flow_rule.ip.val.dst_ip = inet_addr(SOURCE_IP_ADDRESS);
    flow_rule.ip.mask.dst_ip = 0xFFFFFFFF;
    flow_rule.ip.val.src_ip = inet_addr(DESTINATION_IP_ADDRESS);
    flow_rule.ip.mask.src_ip = 0xFFFFFFFF;

    // 9.1.4 Set L4 IP Layer flow rules
    flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
    flow_rule.udp.size = sizeof(flow_rule.udp);
    flow_rule.udp.val.dst_port = htons(UDP_PORT);
    flow_rule.udp.mask.dst_port = 0xFFFF;

    /* 9.2. Attach steering rule to qp. You can attach multiple flows to the same qp - this allows you 
     * to, for example, receive data from multiple IP addresses or in the MeerKAT case, subscribe to multiple multicast 
     * streams.
     */
    struct ibv_flow *eth_flow;
    eth_flow = ibv_create_flow(qp, &flow_rule.attr);
    if (!eth_flow) {
        printf("Couldn't attach steering flow\n");
        exit(1);
    }
    printf("Initialisation Complete - Checking for received packets\n");

    /* 10. Add WRs to the QP - When the NIC receives a packet, it does not automatically know where to put the packets.
     * The CPU needs to give this information to the NIC. This is done by posting Work Requests(WRs) to the NIC. The WR
     * is a struct that specifies what the NIC must do with received data. The WR in turn points to a Scatter Gather 
     * Entry(SGE) struct. This SGE struct gives a pointer to the NIC specifying where in memory a received packet must 
     * be placed. The WR is put on the Receive Queue(RQ) portion of the QP. A single WR is consumed per received packet,
     * as such, the RQ must be periodically receiving new WRs from the CPU. Initially we fill up the RQ. Later on in the
     * code, the new WRs will be posted once a WC is received.
     * 
     * In ibverbs_tx.c, multiple WRs are linked together and posted at once. This has not been done here. This does not
     * mean it cant be done.
     * 
     * TODO: In ibverbs_tx.c, the WR opcode is specified(wr.opcode = IBV_WR_SEND). This has not been done here. I am not
     * sure if specifying this opcode does anything.
     */
    struct ibv_sge sg_entry;
    struct ibv_recv_wr wr, *bad_wr;

    /* pointer to packet buffer size and memory key of each packet buffer */
    sg_entry.length = ENTRY_SIZE;
    sg_entry.lkey = mr->lkey;

    wr.num_sge = 1;
    wr.sg_list = &sg_entry;
    wr.next = NULL;

    for (int n = 0; n < RQ_NUM_DESC; n++)
    {
        /* each descriptor points to max MTU size buffer */
        sg_entry.addr = (uint64_t)pDataBuffer + ENTRY_SIZE*n;
        /* index of descriptor returned when packet arrives */
        wr.wr_id = n;
        /* post WR to ring */
        iReturnValue = ibv_post_recv(qp, &wr, &bad_wr);
        if (iReturnValue) 
        {
            printf("failed to post work request to receive queue %d %d\n",iReturnValue,errno);
            exit(1);
        }
    }

    /* 11. This section is the actual running part of the program. It runs continuously  */
    uint64_t u64NumMessagesComplete;
    struct ibv_wc wc;

    
    struct timeval sTimerStartTime;
    struct timeval sInitialStartTime;
    struct timeval sCurrentTime;

    uint64_t u64PreviousDatagramPayloadPacketIndex=0;
    uint64_t u64CurrentDatagramPayloadPacketIndex=0;
    uint64_t u64NumPacketDrops = 0;
    uint64_t u64NumPacketsReceived = 0;

    gettimeofday(&sInitialStartTime,NULL);
    gettimeofday(&sTimerStartTime,NULL);

    uint64_t u64StartPostSendCount = 0;
    uint64_t u64CurrentPostSendCount = 0;

    uint8_t u8FirstPacket = 0;

    while(1) 
    {
        /* 11.1 Wait for a completion - this is non-blocking and returns 0 most of the time. A completion occurs once a 
         * packet has been received and copied into a the designated memory region. A Work Completion (WC) is posted to 
         * the completion queue when this occurs. 
         */
        u64NumMessagesComplete = ibv_poll_cq(cq_recv, 1, &wc);
        if (u64NumMessagesComplete > 0) 
        {
            /* The address is stored in the sg entry as it will be placed back on the recv queue - it saves creating
             * another variable.
             */
            sg_entry.addr = (uint64_t)pDataBuffer + wc.wr_id*ENTRY_SIZE;
            struct network_packet * p_network_packet = (struct network_packet *)sg_entry.addr;

            /* 11.2 Get the first 8 bytes of the UDP datagram payload which contain the packet index as set by the 
             * transmitter.
             */
            u64CurrentDatagramPayloadPacketIndex = *(uint64_t*) &p_network_packet->udp_datagram_payload;
            u64NumPacketsReceived++;

            /* 11.3 Check that we are not missing a packet. Note, the first packet to be received will always be
             * counted as a dropped packet so I have explicitly excluded this in the missing packet calculation  
             */
            uint64_t u64PacketIndexDiff = u64CurrentDatagramPayloadPacketIndex - u64PreviousDatagramPayloadPacketIndex;   
            if(u64PacketIndexDiff != 1)
            {
                if(u8FirstPacket != 0)
                {
                    u64NumPacketDrops += u64PacketIndexDiff - 1;
                }
                else
                {
                    u8FirstPacket = 1;
                }
                
            }
            u64PreviousDatagramPayloadPacketIndex = u64CurrentDatagramPayloadPacketIndex;

            /* 14.4  After we have this data we need to post a WR back on the receive buffer. The wc.wr_id lets us know
             * which work request was processed which lets us calculate what location in the buffer the packet will be
             * stored in (note: this is applicable to the "sg_entry.addr = (uint64_t)pDataBuffer + wc.wr_id*ENTRY_SIZE"
             * line above.)
             */
            wr.wr_id = wc.wr_id;
            ibv_post_recv(qp, &wr, &bad_wr);
        } 
        else if (u64NumMessagesComplete < 0) 
        {
            printf("Polling error\n");
            exit(1);

        }

        /* If a set number of packets have been received, print information to the screen. I would have preferred to 
         * have waited a set amount of time, but calling gettimeofday() repeatedly resulted in dropped a packets. The
         * 30000000 packet threshold was chosen to correspond to roughly 1 print every 10s at a 100 Gbps data rate.
         */
        if(u64NumPacketsReceived % 30000000 == 0){
            //Calculate data rates
            gettimeofday(&sCurrentTime,NULL);
            double dTimeDifference = (double)sCurrentTime.tv_sec + ((double)sCurrentTime.tv_usec)/1000000.0
                        - (double)sTimerStartTime.tv_sec - ((double)sTimerStartTime.tv_usec)/1000000.0;

            u64CurrentPostSendCount = u64NumPacketsReceived;
            double dDataReceived_Gb = (u64CurrentPostSendCount - u64StartPostSendCount) 
                    * sizeof(struct network_packet)/1000000000 * 8;
            double dDataRate_Gbps = dDataReceived_Gb/dTimeDifference;
            double dTotalDataTransferred_GB = u64NumPacketsReceived * sizeof(struct network_packet)/1000000000;
            double dRuntime_s = (double)sCurrentTime.tv_sec + ((double)sCurrentTime.tv_usec)/1000000.0
                            - (double)sInitialStartTime.tv_sec - ((double)sInitialStartTime.tv_usec)/1000000.0;
            printf("\rRunning Time: %.2fs. Total Received %.3f GB.\
                    Current Data Rate: %.3f Gbps, Packets: received/dropped %ld/%ld",
                    dRuntime_s,dTotalDataTransferred_GB,dDataRate_Gbps, u64NumPacketsReceived, u64NumPacketDrops);
            fflush(stdout);

            //Set timer up for next timing interval.
            u64StartPostSendCount = u64CurrentPostSendCount;
            sTimerStartTime = sCurrentTime;
        }

    }

    //12. This never actually gets called but is left here for completeness.
    printf("Cleanup\n");

    ibv_destroy_flow(eth_flow);
    ibv_dereg_mr(mr);
    free(pDataBuffer);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq_recv);
    ibv_dealloc_pd(pd);
    
    printf("Done\n");

}

