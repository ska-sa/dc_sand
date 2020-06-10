#!/bin/bash
#
# @file     delay_test.sh
#
# @brief    Script that runs multiple different network tests with varying deadtimes to measure how increase deadtime 
#           reduces packet overlap. Clients must be running the \ref repeat_send.sh scripts.
#
# @author   Gareth Callanan
#           South African Radio Astronomy Observatory(SARAO)

deadtime_values_us="0 100 200 300 500 750 1000 1250 1500 2000 2500 3000 4000 5000"
window_length_us=10000
clients=3
process_priority=-19
num_tests=10

for i in $deadtime_values_us
do  
    echo "Look at this value $i $window_length_us"
    sudo chrt 50 numactl -N 0 -C 0 ./udp_receive -t ${clients} -n ${num_tests} -d $i -w ${window_length_us} -o DelayTest_N${num_tests}_W${window_length_us}_D${i}_T${clients} -p
done