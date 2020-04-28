#!/bin/bash
# Script to run the test_bandwidth program multiple times for multiple configurations.
# This script runs a memory bandwidth test with no PCIe transactions. It then adds PCIe transfers to the bandwidth tests.
# The first PCIe test will transfer data to 1 GPU at maximum speed, the second will transfer to two GPUs, etc. This continues until transfers to all GPUs take place simultaneuosly.
# For even finer grained analyses, three different PCIe transfer types are tested - unidirectional host to device transfers, unidirectional device to host transfers and full-duplex bidirectional transfers .
# Each time test_bandwidth is run, the output is piped to a new .csv file. This file has a unique name to make it easy to identify what parameters the test launched with.

numGpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Running Full Suite of Bandwidth Tests"
echo "$numGpus Nvidia Device(s) detected"
PCIeDirFlags="-s -d"
SecondsPerTest=60
MaxThreads=50

timestamp=$(date +"%Y%m%d_%H%M")
gpuMask_dec=0
for i in `seq 0 $numGpus`
do  
    hostname="$(cat /proc/sys/kernel/hostname)"
    baseName="${hostname}_BWTests_${timestamp}_${i}GPUs"
    echo Performing Test: $baseName
    gpuMask_bin=$(echo "obase=2;$gpuMask_dec" | bc)
    if [[ $i -ne 0 ]]; then 
        ./test_bandwidth -g  $gpuMask_bin $PCIeDirFlags -t $SecondsPerTest -b -m 0 -M $MaxThreads -c | tee ${baseName}_pcie_bidir.csv
        ./test_bandwidth -g  $gpuMask_bin -d -t $SecondsPerTest -b -m 0 -M $MaxThreads -c | tee ${baseName}_pcie_D2H.csv
        ./test_bandwidth -g  $gpuMask_bin -s -t $SecondsPerTest -b -m 0 -M $MaxThreads -c | tee ${baseName}_pcie_H2D.csv
    else
        ./test_bandwidth -g  $gpuMask_bin $PCIeDirFlags -t $SecondsPerTest -b -m 0 -M $MaxThreads -c | tee ${baseName}.csv
    fi
    gpuMask_dec=$((($gpuMask_dec<<1)+1))
done