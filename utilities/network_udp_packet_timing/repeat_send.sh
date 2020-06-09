#!/bin/bash
#
# @file     repeat_send.sh
#
# @brief    Script that reruns the client everytime that they finish a test to allow clients to be used in multiple 
#           tests.
#
# @author   Gareth Callanan
#           South African Radio Astronomy Observatory(SARAO)

while :
do
    sudo nice -n -19 numactl -C 0 -m 0 ./udp_send
    sleep 10
done