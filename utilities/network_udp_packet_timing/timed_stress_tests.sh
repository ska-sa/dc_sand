#!/bin/bash

window_length_us=5000
clients=3
process_priority=-19
num_tests=1000
delay_us=1500

for i in `seq 1 10`
do  
    timestamp=$(date +"%Y%m%d_%H%M")
    echo $timestamp
    sudo nice -n -19 numactl -N 0 -C 0 ./udp_receive -t ${clients} -n ${num_tests} -d ${delay_us} -w ${window_length_us} -o StressTest_${timestamp}_N${num_tests}_W${window_length_us}_D${delay_us}_T${clients} -p
    sleep 1800 #30 minutes
done