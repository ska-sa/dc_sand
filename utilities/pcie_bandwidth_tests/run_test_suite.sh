#!/bin/bash
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