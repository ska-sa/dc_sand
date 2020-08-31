# Ethernet Network Acceleration Using ibverbs RDMA library

## Quick Summary

## Terminology

1. RDMA
2. ibverbs
3. InfiniBand

## Recommended Reading

1. [The Fundamentals of RDMA Programming](https://academy.mellanox.com/en/course/rdma-programming-intro/?cm=446)

2. [RDMA Aware Programming User Manual](https://www.mellanox.com/related-docs/prod_software/RDMA_Aware_Programming_user_manual.pdf)

3. [example](https://community.mellanox.com/s/article/raw-ethernet-programming--basic-introduction---code-example)

4. [Man Page](https://man7.org/linux/man-pages/man3/ibv_create_flow.3.html)

5. [Mellanox OFED Manual](https://docs.mellanox.com/display/MLNXOFEDv461000/Ethernet+Network)

6. [Checksum Offload](https://manpages.debian.org/testing/libibverbs-dev/ibv_post_send.3.en.html)

## Compiling and running the example

Requires Mellanox OFED Driver
Tested on Mellanox ConnectX-5 NIC 
Need to be sudo
Easiest to test with tcpdump: sudo tcpdump -i < interface name> -vvvv -X -s 256 port 7708 -c 1
Hard coded arp -> just run `arp`(on our servers by default) command. If the device does not show up, ping it, or set mac address to appropriate gateway if L3 network

### Test