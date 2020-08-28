#include <infiniband/arch.h>
#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
/* Functions in this library are used for converting IP address strings to character arrays and for converting data 
 * between network and host byte order.
 */
#include <arpa/inet.h> 
#include <sys/time.h>   //For timing functions

#include "common_functions.h"

#define RQ_NUM_DESC 2048
#define ENTRY_SIZE 9000 /* The maximum size of each received packet - set to jumbo frame */
#define LOCAL_INTERFACE_IP_ADDRESS "10.100.18.9"
#define REMOTE_INTERFACE_IP_ADDRESS "10.100.18.7"
#define UDP_PORT 7708

int main()
{
    struct ibv_device *ib_dev;
    struct ibv_context *context;
    struct ibv_pd *pd;
    uint8_t u8PortNum;

    int ret;

    /* 1. Get Device */
    ib_dev = get_ibv_device_from_ip(&u8PortNum, LOCAL_INTERFACE_IP_ADDRESS);
    if(ib_dev == NULL){
        printf("No NIC with matching SOURCE_IP_ADDRESS found\n");
        exit(1);
    }

    /* 2. Get the device context */
    /* Get context to device. The context is a descriptor and needed for resource tracking and operations */
    context = ibv_open_device(ib_dev);
    if (!context) {
        printf("Couldn't get context for %s\n",ibv_get_device_name(ib_dev));
        exit(1);
    }

    /* 3. Allocate Protection Domain */
    /* Allocate a protection domain to group memory regions (MR) and rings */
    pd = ibv_alloc_pd(context);
    if (!pd) {
        printf("Couldn't allocate PD\n");
        exit(1);
    }

    /* 4. Create Complition Queue (CQ) */
    struct ibv_cq *cq_recv;
    cq_recv = ibv_create_cq(context, RQ_NUM_DESC, NULL, NULL, 0);

    if (!cq_recv) {
        printf("Couldn't create cq_recv\n");
        exit (1);
    }

    /* 5. Initialize QP */
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

    /* 6. Create Queue Pair (QP) - Receive Ring */
    qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
        printf("Couldn't create RSS QP\n");
        exit(1);
    }

    /* 7. Initialize the QP (receive ring) and assign a port */
    struct ibv_qp_attr qp_attr;
    int qp_flags;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_flags = IBV_QP_STATE | IBV_QP_PORT;
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.port_num = u8PortNum; //I have never had this value equal to anything other than 1, I have a niggling concern that if it equals another number things will not work;
    ret = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (ret) {
        printf("failed modify qp to init %d %d\n",ret, errno);
        exit(1);
    }
    memset(&qp_attr, 0, sizeof(qp_attr));

    /* 8. Move ring state to ready to receive, this is needed in order to be able to receive packets */
    qp_flags = IBV_QP_STATE;
    qp_attr.qp_state = IBV_QPS_RTR;
    ret = ibv_modify_qp(qp, &qp_attr, qp_flags);
    if (ret) {
        printf("failed modify qp to receive %d %d\n",ret,errno);
        exit(1);
    }

    /* 9. Allocate Memory */
    int buf_size = ENTRY_SIZE*RQ_NUM_DESC; /* maximum size of data to be accessed by hardware */
    void *buf;
    buf = malloc(buf_size);
    if (!buf) {
        printf("Couldn't allocate memory\n");
        exit(1);
    }

    /* 10. Register the user memory so it can be accessed by the HW directly */
    struct ibv_mr *mr;
    mr = ibv_reg_mr(pd, buf, buf_size, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) {
        printf("Couldn't register mr\n");
        exit(1);
    }

    /* 11. Attach all buffers to the ring */
    int n;
    struct ibv_sge sg_entry;
    struct ibv_recv_wr wr, *bad_wr;

    /* pointer to packet buffer size and memory key of each packet buffer */
    sg_entry.length = ENTRY_SIZE;
    sg_entry.lkey = mr->lkey;

    /*
     * descriptor for receive transaction - details:
     * - how many pointers to receive buffers to use
     * - if this is a single descriptor or a list (next == NULL single)
     */

    wr.num_sge = 1;
    wr.sg_list = &sg_entry;
    wr.next = NULL;

    for (n = 0; n < RQ_NUM_DESC; n++) {
        /* each descriptor points to max MTU size buffer */
        sg_entry.addr = (uint64_t)buf + ENTRY_SIZE*n;
        /* index of descriptor returned when packet arrives */
        wr.wr_id = n;
        /* post receive buffer to ring */
        ret = ibv_post_recv(qp, &wr, &bad_wr);
        if (ret) {
            printf("failed to post work request to receive queue %d %d\n",ret,errno);
            exit(1);
        }
    }

    /* 12. Register steering rule to intercept packet to DEST_MAC and place packet in ring pointed by ->qp */
    
    //Why do I declare the stuct here? Because thats how everyone does it in the docs.
    struct
    {
        struct ibv_flow_attr attr;
        struct ibv_flow_spec_eth eth __attribute__((packed));
        struct ibv_flow_spec_ipv4 ip __attribute__((packed));
        struct ibv_flow_spec_tcp_udp udp __attribute__((packed));
    } __attribute__((packed)) flow_rule;
    memset(&flow_rule, 0, sizeof(flow_rule));

    //General Rule Set Up (IBV_FLOW_ATTR_SNIFFER)
    flow_rule.attr.type = IBV_FLOW_ATTR_NORMAL;
    flow_rule.attr.priority = 0;
    flow_rule.attr.size = sizeof(flow_rule);
    flow_rule.attr.num_of_specs = 3;
    flow_rule.attr.port = 1;

    //Set L2 Layer flow rules
    flow_rule.eth.type = IBV_FLOW_SPEC_ETH;
    flow_rule.eth.size = sizeof(flow_rule.eth);

    //Set L3 IP Layer flow rules
    flow_rule.ip.type = IBV_FLOW_SPEC_IPV4;
    flow_rule.ip.size = sizeof(flow_rule.ip);
    flow_rule.ip.val.dst_ip = inet_addr(LOCAL_INTERFACE_IP_ADDRESS);
    flow_rule.ip.mask.dst_ip = 0xFFFFFFFF;
    flow_rule.ip.val.src_ip = inet_addr(REMOTE_INTERFACE_IP_ADDRESS);
    flow_rule.ip.mask.src_ip = 0xFFFFFFFF;

    //Set L4 IP Layer flow rules
    flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
    flow_rule.udp.size = sizeof(flow_rule.udp);
    flow_rule.udp.val.dst_port = htons(UDP_PORT);
    flow_rule.udp.mask.dst_port = 0xFFFF;

    /* 13. Attach steering rule to qp*/
    struct ibv_flow *eth_flow;
    eth_flow = ibv_create_flow(qp, &flow_rule.attr);
    if (!eth_flow) {
        printf("Couldn't attach steering flow\n");
        exit(1);
    }
    printf("Initialisation Complete - Checking for received packets\n");

    /* 14. Wait for CQ event upon message received, and print a message */
    int msgs_completed;
    struct ibv_wc wc;

    
    struct timeval sTimerStartTime;
    struct timeval sInitialStartTime;
    struct timeval sCurrentTime;

    uint64_t u64PreviousDatagramPayloadPacketIndex=0;
    uint64_t u64CurrentDatagramPayloadPacketIndex=0;
    uint64_t u64NoPacketDropInterval = 0;
    uint64_t u64NumPacketDrops = 0;
    uint64_t u64NumPacketsReceived = 0;

    gettimeofday(&sInitialStartTime,NULL);
    gettimeofday(&sTimerStartTime,NULL);

    uint64_t u64StartPostSendCount = 0;
    uint64_t u64CurrentPostSendCount;

    while(1) 
    {
        /* wait for completion */
        msgs_completed = ibv_poll_cq(cq_recv, 1, &wc);
        if (msgs_completed > 0) {
            /*
            * completion includes:
            * -status of descriptor
            * -index of descriptor completing
            * -size of the incoming packets
            */

            sg_entry.addr = (uint64_t)buf + wc.wr_id*ENTRY_SIZE;
            struct network_packet * p_network_packet = (struct network_packet *)sg_entry.addr;
            u64CurrentDatagramPayloadPacketIndex = *(uint64_t*) &p_network_packet->udp_datagram_payload;
            u64NumPacketsReceived++;

            if(u64PreviousDatagramPayloadPacketIndex == 0)
            {
                u64PreviousDatagramPayloadPacketIndex = u64CurrentDatagramPayloadPacketIndex -1;
            }
            else
            {
                uint64_t u64PacketIndexDiff = u64CurrentDatagramPayloadPacketIndex - u64PreviousDatagramPayloadPacketIndex;
                //printf("message %ld received size %d\n", wc.wr_id, wc.byte_len);    
                if(u64PacketIndexDiff != 1)
                {
                    u64NumPacketDrops += u64PacketIndexDiff - 1;
                    u64NoPacketDropInterval = 0;
                }
                u64PreviousDatagramPayloadPacketIndex = u64CurrentDatagramPayloadPacketIndex;
            }

            wr.wr_id = wc.wr_id;
            /* after processed need to post back buffer */
            ibv_post_recv(qp, &wr, &bad_wr);
            u64NoPacketDropInterval++;
            
            //gettimeofday(&sCurrentTime,NULL);
        } 
        else if (msgs_completed < 0) 
        {
            printf("Polling error\n");
            exit(1);

        }

        //Measure time and if a second has passed print the data rate to screen.
        //
        
        if(u64NumPacketsReceived % 30000000 == 0){
            //Calculate data rate
            gettimeofday(&sCurrentTime,NULL);
            double dTimeDifference = (double)sCurrentTime.tv_sec + ((double)sCurrentTime.tv_usec)/1000000.0
                        - (double)sTimerStartTime.tv_sec - ((double)sTimerStartTime.tv_usec)/1000000.0;

            u64CurrentPostSendCount = u64NumPacketsReceived;
            double dDataReceived_Gb = (u64CurrentPostSendCount - u64StartPostSendCount) * sizeof(struct network_packet)/1000000000 * 8;
            double dDataRate_Gbps = dDataReceived_Gb/dTimeDifference;
            double dTotalDataTransferred_GB = u64NumPacketsReceived * sizeof(struct network_packet)/1000000000;
            double dRuntime_s = (double)sCurrentTime.tv_sec + ((double)sCurrentTime.tv_usec)/1000000.0
                            - (double)sInitialStartTime.tv_sec - ((double)sInitialStartTime.tv_usec)/1000000.0;
            printf("\rRunning Time: %.2fs. Total Received %.3f GB. Current Data Rate: %.3f Gbps, Packets: received/dropped %ld/%ld",dRuntime_s,dTotalDataTransferred_GB,dDataRate_Gbps, u64NumPacketsReceived, u64NumPacketDrops);
            fflush(stdout);

            //Set timer up for next second
            u64StartPostSendCount = u64CurrentPostSendCount;
            sTimerStartTime = sCurrentTime;
        }

    }

    printf("Cleanup\n");

    ibv_destroy_flow(eth_flow);
    ibv_dereg_mr(mr);
    free(buf);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq_recv);
    ibv_dealloc_pd(pd);
    
    printf("Done\n");

}

