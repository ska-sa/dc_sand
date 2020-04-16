In order to enable huge pages on an Ubuntu system run the following commands:
sudo sysctl -w vm.nr_hugepages=30000

To check this works run the command:
cat /proc/meminfo | grep Huge
