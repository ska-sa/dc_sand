# Introduction

TODO

# Installation

TODO

# Using the Utility

TODO

# Maximising Performance

TODO

## Useful Network Monitoring Commands
1. ethtool
    1. `sudo ethtool -c enp175s0d1` - View interrupt coalescing settings.
    2. `sudo ethtool -a enp175s0d1` - Determine if link flow control is enabled.
    3. `sudo ethtool -g enp175s0d1` - View network buffer sizes.
    4. `sudo ethtool -S enp175s0d1` - View all NIC counters. Including TX/RX counters.
2. `sudo netstat -neopa` - Display all process to socket/ip/port mappings.
3. netstat
    1. `netstat -su` - Display all protocol specific counters. Does not seperate by interface.
    2. `netstat -i` - Abridged display showing a few basic counters for each interface, including tx/rx packets 
    recieved and tx/rx errors.

## Ethtool NIC configuration commands

1. `ethtool -G enp175s0d1 rx 8192 tx 8192` - Set network buffer sizes to prevent packets getting dropped due to buffer 
    overflow. 8192 bytes is the maximum size on most current NICs.
2. `sudo ethtool -C enp175s0d1 adaptive-tx off tx-usecs 16 tx-frames 16` - Set interrupt coalescing for TX parameters. 
    Setting both usecs and frames to 16 seems to be the optimal setup on dbelab07. 
3. `sudo ethtool -C enp175s0d1 adaptive-rx off rx-usecs 0 rx-frames 0` - Set interrupt coalescing for RX parameters. 
    Setting receiver side rx-usecs and rx-frames to zero has resulted in the least packets being dropped by the 
    receiver. This is specifically in the 10 Gbps domain. It is suspected that this may not hold up at the 100 Gbps 
    rates.

Here is a [link](https://access.redhat.com/sites/default/files/attachments/20150325_network_performance_tuning.pdf) to
a useful guide on tuning network performance. It provides details the reasons for most of the above commands.

## Processor C-states
Modern processor can be in different modes known as C-states. The C-states reduce the CPU power consumption by putting 
the CPU in hibernation mode. Waking up to service a received packet can results in packet drops. C0 is indicates the 
CPU runs at maximum performance and power with higher numbers indicating that the CPU spends more time hibernating. 

The following are useful commands when looking at C-states:
1. `cat /sys/module/intel_idle/parameters/max_cstate` - View the maximum system c-state.
2. `sudo powertop` - Detailed system power consumption and C-state levels can be monitored using the 
[powertop](https://01.org/powertop) utility.

C-states can be disabled in system bios or by modifying some grub utilities. C-states have not yet been disabled but 
a reduction in performance when a stream first starts transmitting has been observed and this has been attributed to 
the CPU switching from low power to high power C-states.

#TODO: Disable C-states

## NUMA Boundaries

#TODO: FIll this in

`numactl -N 0 -m 0 ./udp_receive`

not 

`numactl -C 4 -m 0 ./udp_receive`

Check `sudo mst status -v` for NUMA information

## Process Priority
By default the network threads are launched with the same priority. The network threads need to be more responsive to 
ensure maximum rates are achieved at the transmitter and no packets are dropped at the receiver. Increasing the priority
of these threads ensures that they remain running at all times and are not preempted.

The first attempt to increase process priority was to use the `nice` command:

`sudo nice -n -5 ./udp_send` (lower value is higher priority)

However `nice` only modifies userspace priorities. It turns out that Linux has moderate support for soft real time 
applications. The `chrt` utility allows real time priorities to be assigned to processors that have higher priority 
than any userspace and most kernelspace programs:

`sudo chrt 50 ./udp_send` (Higher value is higher priority)

## Timing 

In order to transmit packets at precise time intervals, all nodes on the network need to agree on what the time is.

#TODO

1. ### NTP 
    1. Install NTP if it is not installed : `sudo apt-get install ntp`.
    2. View the ntp servers that are connected: `ntpq -p`. The ip address with the "*" next to it is the one that the 
    server is using to set its time.
    3. Go to the ntp configuration `/etc/ntp.conf` file and add the following line: `server ntp.kat.ac.za iburst`. This 
    line adds the local ntp server to the list of available servers. Removing the other servers will force your server 
    to sync to that one server.
    4. Restart the ntp server: `sudo service ntp restart`. Run `ntpq -p` to confirm that the server now appears in the 
list of available NTP servers and that it has a "*" next to it indicating the time.

2. ### PTP 

## Libvma
Mellanox provides a library called VMA that accelerates the performance of standard socket applications. This is done 
transparently. The user creates ordinary sockets and the vma tools will bypass the standard(slow) kernel and network 
stack, instead implementing the functionality in the userspace using the mellanox verbs api. The full guide can be 
found [here](https://www.mellanox.com/related-docs/prod_acceleration_software/VMA_8_6_10_User_Manual.pdf).

Libvma is relativly simple to use. To implement it:
1. Ensure libvma is is installed. Refer to the 
[installation guide](https://www.mellanox.com/related-docs/prod_acceleration_software/VMA_8_6_10_Installation_Guide.pdf)
.
2. Launch the command line utilities prefaced with "LD_PRELOAD=libvma<span>.so</span>", eg: 
`LD_PRELOAD=libvma.so ./udp_send`

On the udp_send side using `LD_PRELOAD=libvma.so` made a version of the sender go from transmitting at a data rate of 
~12Gbps to a data rate of ~41 Gbps. This occured without modifying any of the source code. By using  
`LD_PRELOAD=libvma.so ./udp_receive`, the receiver also dropped significantly fewer packets. Using this mode, 0 packets
were dropped at the 12 Gbps data rate while 60% of packets were dropped at the 41 Gbps data rate. The network timing 
tests do note require data to be transmitted at full line rate, as such libvma was enabled on the receiver not the 
sender to ensure the most packets were received and analysed.

Additional VMA configuration can be given as arguments after the `LD_PRELOAD` command: \
`LD_PRELOAD=libvma.so VMA-SELECT-POLL=-1 VMA_THREAD_MODE=0 ./udp_receive` \
Additional configuration opetions are specified in the vma user guide as well as a
[VMA Performance Tuning Guide](https://community.mellanox.com/s/article/vma-performance-tuning-guide) on the Mellanox 
website.

Libvma improves performance so significantly on the receiver at 12 Gbps that the `numactl` and `chrt` commands become 
unnecessary on smaller tests. The overnight tests have not been run without `numactl` or `chrt`, so the long term 
stability without these commands is unknown. 

## Putting it all together

#TODO: Describe other configuration commands and seperate them out by receiver/transmitter and once off commands

A number of commands are chained together to ensure the receiver is run as optimally as possible: 
1. Receiver: `sudo LD_PRELOAD=libvma.so VMA-SELECT-POLL=-1 VMA_THREAD_MODE=0 VMA_SPEC=latency chrt 50 numactl 
-N 0 -m 0 ./udp_receive -t 1 -n 100 -d 500 -w 100000 -p -o DelayTest`
2. Transmitter: `sudo LD_PRELOAD=libvma.so chrt 50 numactl -m 0 -C 4 ./udp_send`

