#!/bin/bash
while :
do
    sudo nice -n -19 numactl -C 0 -m 0 ./udp_send
    sleep 10
done