## Useful Network Monitoring Commands:
1. ethtool
    1. `sudo ethtool -c enp175s0d1` - View interrupt coalescing settings.
    2. `sudo ethtool -C enp175s0d1 tx-usecs 16 tx-frames 16` - Set interrupt coalescing for TX parameters. Setting both\
    usecs and frames to 16 seems to be the optimal setup on dbelab07.
    3. `sudo ethtool -a enp175s0d1` - Determine if link flow control is enabled.
    4. `sudo ethtool -g enp175s0d1` - View network buffer sizes.
    5. `ethtool -G enp175s0d1 rx 8192 tx 8192` - Set network buffer sizes to prevent packets getting dropped due to \
    buffer overflow.
    6. `sudo ethtool -S enp175s0d1` - View all NIC counters. Including TX/RX counters.
2. `sudo netstat -neopa` - Display all process to socket/ip/port mappings.
3. netstat
    1. `netstat -su` - Display all protocol specific counters. Does not seperate by interface.
    2. `netstat -i` - Abridged display showing a few basic counters for each interface, including tx/rx packets recieved \
    and tx/rx errors.

## Processor C-states
Modern processor can be in different modes known as C-states. The C-states reduce the CPU power consumption by putting \
the CPU in hibernation mode. Waking up to service a received packet can results in packet drops. C0 is indicates the CPU \
runs at maximum performance and power with higher numbers indicating that the CPU spends more time hibernating. 

The following are useful commands when looking at C-states:
1. `cat /sys/module/intel_idle/parameters/max_cstate` - View the maximum system c-state.
2. `sudo powertop` - Detailed system power consumption and C-state levels can be monitored using the [powertop](https://01.org/powertop) utility.

C-states can be disabled in system bios.

## Process Priority
`sudo nice -n -5 ./udp_send` (lower value is higher priority)

`sudo chrt 50 numactl -C 0 -m 0 ./udp_send` (Higher value is higher priority)

## NTP 
1. Install NTP if it is not installed : `sudo apt-get install ntp`.
2. View the ntp servers that are connected: `ntpq -p`. The ip address with the "*" next to it is the one that the server
is using to set its time.
3. Go to the ntp configuration `/etc/ntp.conf` file and add the following line: `server ntp.kat.ac.za iburst`. This line
adds the local ntp server to the list of available servers. Removing the other servers will force your server to sync to 
that one server.
4. Restart the ntp server: `sudo service ntp restart`. Run `ntpq -p` to confirm that the server now appears in the list 
of available NTP servers and that it has a "*" next to it indicating the time

## Additional Reading
Here is a [link](https://access.redhat.com/sites/default/files/attachments/20150325_network_performance_tuning.pdf) to a useful guide on tuning network performance. It provides details the reasons for most of the above \
commands.
