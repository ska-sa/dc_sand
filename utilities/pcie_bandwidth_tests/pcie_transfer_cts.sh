#!/bin/bash
#Script for continously transferring data to the PCIe devices. Will output rate to terminal every 20 seconds.

echo "PCIe Continous Transfer Script"

if [ "$#" -eq 0 ]; then
    numGpus=1
else
    numGpus=$1
fi
echo "Performing transfers on $numGpus GPUs or maximum number of installed GPUs, whichever is lower"

gpuMask=""
for i in `seq 0 $numGpus`
do  
    gpuMask=${gpuMask}1
done

while :
do
	./test_bandwidth -g $gpuMask -d -s -t 20 -c
done