#!/bin/bash

delay_values_us="0 100 200 300 500 750 1000 1250 1500 2000 2500 3000 4000 5000"
window_length_us=10000
clients=3
process_priority=-19
num_tests=10

for i in $delay_values_us
do  
    echo "Look at this value $i $window_length_us"
    sudo nice -n -19 numactl -N 0 -C 0 ./udp_receive -t ${clients} -n ${num_tests} -d $i -w ${window_length_us} -o DelayTest_N${num_tests}_W${window_length_us}_D${i}_T${clients} -p
done