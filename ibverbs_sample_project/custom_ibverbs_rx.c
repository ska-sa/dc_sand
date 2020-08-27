//#include <rdma/rdma_cma.h>
#include <infiniband/arch.h>
#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <arpa/inet.h>

#define RQ_NUM_DESC 512
#define ENTRY_SIZE 9000 /* The maximum size of each received packet - set to jumbo frame */
#define LOCAL_INTERFACE "10.100.18.9"
#define REMOTE_INTERFACE "10.100.18.7"
#define UDP_PORT 7708

int main()
{

    struct ibv_device **dev_list;
    struct ibv_device *ib_dev;
    struct ibv_context *context;
    struct ibv_pd *pd;

    int ret;
     
    /* Get the list of offload capable devices */
    dev_list = ibv_get_device_list(NULL);
    if (!dev_list) {
        perror("Failed to get IB devices list");
        exit(1);
    }

    /* 1. Get Device */
    /* In this example, we will use the first adapter (device) we find on the list (dev_list[0]) . You may change the code in case you have a setup with more than one adapter installed. */
    ib_dev = dev_list[1];
    printf("RDMA device[%d]: name=%s\n", 1, ibv_get_device_name(dev_list[1]));
    if (!ib_dev) {
        printf("IB device not found\n");
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
    qp_attr.port_num = 1;
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
    flow_rule.ip.val.dst_ip = inet_addr(LOCAL_INTERFACE);
    flow_rule.ip.mask.dst_ip = 0xFFFFFFFF;
    flow_rule.ip.val.src_ip = inet_addr(REMOTE_INTERFACE);
    flow_rule.ip.mask.src_ip = 0xFFFFFFFF;

    //Set L4 IP Layer flow rules
    flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
    flow_rule.udp.size = sizeof(flow_rule.udp);
    flow_rule.udp.val.dst_port = htons(UDP_PORT);
    flow_rule.udp.mask.dst_port = 0xFFFF;

    /* 13. Create steering rule */

    struct ibv_flow *eth_flow;
    eth_flow = ibv_create_flow(qp, &flow_rule.attr);
    if (!eth_flow) {
        printf("Couldn't attach steering flow\n");
        exit(1);
    }

    // This was just a test to see the flags supported by the device.
    // struct ibv_device_attr device_attr;
    // ibv_query_device(context, &device_attr);
    // printf("Normal: 0x%08x\n",device_attr.device_cap_flags);
    // int num1 = device_attr.device_cap_flags;

    // struct ibv_exp_device_attr attr;
    // ibv_exp_query_device(context, &attr);
    // printf("Experimental: 0x%08x\n",attr.exp_device_cap_flags);
    // int num2 = attr.exp_device_cap_flags;

    // for (size_t i = 0; i < 32; i++)
    // {
    //     int lastPos1 = 0x1 & num1;
    //     int lastPos2 = 0x1 & num2;
    //     printf("%ld Norm %d Exp %d\n",i,lastPos1, lastPos2);
    //     num1 = num1 >> 1;
    //     num2 = num2 >> 1;
    // }

    printf("Initialisation Complete - Checking for received packets\n");

    /* 14. Wait for CQ event upon message received, and print a message */
    int msgs_completed;
    struct ibv_wc wc;

    //sleep(10);

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
        printf("message %ld received size %d\n", wc.wr_id, wc.byte_len);
        sg_entry.addr = (uint64_t)buf + wc.wr_id*ENTRY_SIZE;
        wr.wr_id = wc.wr_id;
        
        /* after processed need to post back buffer */
        ibv_post_recv(qp, &wr, &bad_wr);
        
        } 
        else if (msgs_completed < 0) 
        {
            printf("Polling error\n");
            exit(1);

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

//#define LOCAL_INTERFACE "10.100.18.9"
//#define PORT 20069

//NOTE: A completion vector seems to be a mapping of the IRQs from the RDAM NIC to a specific CPU

// int main(int argc, char *argv[]){
//     int status;
//     struct ibv_device *ib_dev;
//     struct ibv_device **dev_list;
//     printf("RDMA Test\n");
//     //struct rdma_event_channel *event_channel = rdma_create_event_channel();
//     printf("RDMA Event Channel Created\n");

//     //Part 1
//     // struct rdma_cm_id *cm_id = malloc(sizeof(struct rdma_cm_id));
//     // int status = rdma_create_id(event_channel, &cm_id, NULL, RDMA_PS_UDP); //Should this be NULL
//     // if (status < 0){
//     //     printf("rdma_create_id failed\n");
//     //     return 1;
//     // }
//     // printf("RDMA CM ID created: %d\n",status);

//     dev_list = ibv_get_device_list(NULL);
//     ib_dev = dev_list[1];
//     printf("RDMA device[%d]: name=%s\n", 1, ibv_get_device_name(dev_list[1]));
//     if (!ib_dev) {
//         fprintf(stderr, "IB device not found\n");
//         exit(1);
//     }

//     //Part 2
//     // struct sockaddr_in *sin1 = malloc(sizeof(struct sockaddr));
//     // sin1->sin_family = AF_INET; 
//     // sin1->sin_port = htons(PORT);
//     // sin1->sin_addr.s_addr = inet_addr(LOCAL_INTERFACE);//INADDR_ANY;
    
//     // status = rdma_bind_addr(cm_id, (struct sockaddr *)sin1);
// 	// if (status) {
// 	// 	printf("rdma_bind_addr failed\n");
// 	// 	return 1;
// 	// }
// 	// printf("rdma_bind_addr successful\n");

//     struct ibv_context *context = ibv_open_device(ib_dev);
//     if (!context){
//         printf("ibv_open_device failed\n");
//         return 1;
//     }
//     printf("ibv_open_device successful\n");
    
//     //Step 3
//     //struct ibv_pd *pd = ibv_alloc_pd(cm_id->verbs);
//     struct ibv_pd *pd = ibv_alloc_pd(context);
//     if (!pd){
//         printf("ibv_alloc_pd failed\n");
//         return 1;
//     }
//     printf("ibv_alloc_pd successful\n");

//     //Step 4
//     //struct ibv_comp_channel *comp_channel = ibv_create_comp_channel(cm_id->verbs);
//     // struct ibv_comp_channel *comp_channel = ibv_create_comp_channel(context);
//     // if (!comp_channel)
//     // {
//     //     printf("ibv_create_comp_channel failed\n");
//     //     return 1;
//     // }
//     // printf("ibv_create_comp_channel succesful\n");

//     //Step 5
//     int iNumCqe = 512; //Number of completion queue entries to be held in the completion queue, not sure on the order of magnitude for this queue
//     int iCompVector = 1; //Some mapping of IRQs to specific cores, needs to be investigated
//     //struct ibv_cq *recv_cq = ibv_create_cq(cm_id->verbs, iNumCqe, NULL, comp_channel, iCompVector);//Might need to be ibv_exp_create_cq
//     struct ibv_cq *recv_cq = ibv_create_cq(context, iNumCqe, NULL, NULL, 0);//Might need to be ibv_exp_create_cq
//     //struct ibv_cq *recv_cq = ibv_create_cq(context, iNumCqe, NULL, comp_channel, iCompVector);//Might need to be ibv_exp_create_cq
//     if(!recv_cq)
//     {
//         printf("ibv_create_cq failed\n");
//         return 1;
//     }
//     //if (ibv_req_notify_cq(recv_cq, 0)){
//     //    printf("ibv_req_notify_cq failed\n");
//     //    return 1;
//     //}
//     printf("ibv_create_cq succesful\n");

//     //Step 6 - as far as I can tell this step is only useful to make sure the queue pair object has a seperate send and recv completion queue(cq), the send cq is not actually used
//     //struct ibv_cq *send_cq = ibv_create_cq(cm_id->verbs, 1, NULL, NULL, 0);//Might need to be ibv_exp_create_cq
//     struct ibv_cq *send_cq = ibv_create_cq(context, 1, NULL, NULL, 0);//Might need to be ibv_exp_create_cq
//     if(!send_cq)
//     {
//         printf("ibv_create_cq failed\n");
//         return 1;
//     }
//     //if (ibv_req_notify_cq(send_cq, 0)){
//     //    printf("ibv_req_notify_cq failed\n");
//     //    return 1;
//     //}
//     printf("ibv_create_cq succesful\n");

//     //Step 7
//     struct ibv_qp_init_attr attr;
//     //
//     memset(&attr, 0, sizeof(attr));
//     attr.send_cq = send_cq;
//     attr.recv_cq = recv_cq;
//     attr.qp_type = IBV_QPT_RAW_PACKET;//Bmerry has this set to IBV_QPT_RAW_PACKET, throws an error when I do this.
//     attr.cap.max_send_wr = 1;
//     attr.cap.max_recv_wr = iNumCqe;
//     attr.cap.max_send_sge = 1;
//     attr.cap.max_recv_sge = 1;

//     struct ibv_qp *qp = ibv_create_qp(pd, &attr);
//     //struct ibv_qp *qp = ibv_exp_create_qp(cm_id->verbs, &attr);
//     //status =  rdma_create_qp(cm_id, pd, &attr);
//     if(qp)
//     {
//         printf("ibv_create_qp failed %d\n",status);
//         return 1;
//     }
//     printf("ibv_create_qp succesful\n");

//     //Step 8
//     int iBufSize_bytes = 1000000;
//     uint8_t * buf = calloc(iBufSize_bytes, sizeof(uint8_t));
//     printf("Buffer allocated\n");

//     struct ibv_mr * mr = ibv_reg_mr(pd, buf, iBufSize_bytes * sizeof (uint8_t), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE); //Should it be rdma_reg_memory_region
//     if(!mr)
//     {
//         printf("ibv_reg_mr failed\n");
//         return 1;
//     }
//     printf("ibv_reg_mr succesful\n");
//     //std::vector<ibv_flow_t> flows;
//     //ibv_mr_t mr;


//     printf("Clean Up\n");
//     ibv_dereg_mr(mr);
//     free(buf);
//     //rdma_destroy_qp(cm_id);
//     ibv_destroy_qp(qp);
//     ibv_destroy_cq(recv_cq);
//     ibv_destroy_cq(send_cq);
//     //ibv_destroy_comp_channel(comp_channel);
//     //rdma_destroy_event_channel(event_channel);
//     //rdma_destroy_id(cm_id);
//     //free(sin1);
//     //free(cm_id);
//     printf("Finished\n");
//     return 0;
// }