## Useful Network Monitoring Commands:
1. ethtool
    1. `sudo ethtool -c enp175s0d1` - View interrupt coalescing settings
    2. `sudo ethtool -a enp175s0d1` - Determine if link flow control is enabled.
    3. `sudo ethtool -g enp175s0d1` - View network buffer sizes.
    4. `sudo ethtool -S enp175s0d1` - View all NIC counters. Including TX/RX counters
2. `sudo netstat -neopa` - Display all process to socket/ip/port mappings
3. netstat
    1. `netstat -su` - Display all protocol specific counters. Does not seperate by interface
    2. `netstat -i` - Abridged display showing a few basic counters for each interface, including tx/rx packets recieved \
    and tx/rx errors.

## Processor C-states
Modern processor can be in different modes known as C-states. The C-states reduce the CPU power consumption by putting \
the CPU in hibernation mode. Waking up to service a received packet can results in packet drops. C0 is indicates the CPU \
runs at maximum performance and power with higher numbers indicating that 

The following are useful commands when looking at C-states:
1. `cat /sys/module/intel_idle/parameters/max_cstate` - View the maximum system c-state.
3. `sudo powertop` - Detailed system power consumption and C-state levels can be monitored using the [powertop](https://01.org/powertop) utility.

C-states can be disabled in system bios.

## Additional Reading
Here is a [link]((https://access.redhat.com/sites/default/files/attachments/20150325_network_performance_tuning.pdf)) to a useful guide on tuning network performance. It provides details the reasons for most of the above \
commands.
