# Bandwidth Test Utility

## 1. Description
This utility has been designed to benchmark two parameters:
1. Server Memory Bandwidth - this benchmark uses a function written in assembly to read contiguous sections of memory into CPU AVX registers. Furthermore, it supports multiple threads performing these reads as this has been shown to achieve a higher bandwidth
2. PCIe Transfer Speeds - this measures the maximum rate achievable across the PCIe bus. The user can configure the transfer direction(host to device, device to host or bidirectional) and the number of PCIe cards to transfer simultaneously(Some servers have multiplexed PCIe lanes). At the moment this is done using CUDA and as such only works on CUDA GPUs. The framework is extendable - OpenCL, etc should be easy to add. The utility supports  testing multiple PCIe devices in parallel

The reason this framework exists is because SARAO needed to test how flooding the RAM and PCIe busses simultaneously impacts the max performance of both these busses. As such this utility is heavily multithreaded to support running all these tests in parallel.

## 2. Installation and Running

In order to install this utility navigate to the utilities/pcie_bandwidth_tests directory in the dc_sand repo and run `make`. You may need to install the C++ boost library if you haven't already.

Running `make` will produce a test_program program. Running `./test_bandwidth -h` will give you a list of all available configurations.

**Example**: `Running ./test_bandwidth -g 1 -d -t 60 -b -m 1 -M 10` will run a PCI bandwidth test on GPU0(`-g 1`) in the host to device direction(`-d`). It will also run memory bandwidth tests(`-b`) with an initial thread count of one(`-m 1`) and it will keep increasing the number of threads until ten threads are used(`-M 10`). Each thread test will be run for 60 seconds(`-t 60`) before the number of threads is increased.

The output from the utility should make all the rates clear.

## 3. Performing Bulk Tests
Often one wants to run a suite of tests on a server to examine how the PCIe and RAM bus interact. The [run_test_suite.sh](run_test_suite.sh) automatically does this. It runs a RAM bus only test and then runs another test with 1 GPU, then another with 2 etc. For each number of gpus, three tests are run - one performing H2D transfers, the other D2H transfers and the final bidirectional transfers.

This produces a .csv file for each test. This file has a verbose name so that it is clear

## 4. Enabling Huge Pages
This utility allows for memory to be allocated using huge pages with the `-p` flag. Enabling hugh pages on a server can impact memory bandwidth. On systems running the Ubuntu OS, hugh pages are not enabled by default.

In order to enable huge pages on an Ubuntu system run the following commands:
```
sudo sysctl -w vm.nr_hugepages=30000
```
To check this works run the command:
```
cat /proc/meminfo | grep Huge
```
The output produced by the above command should be similar to this:
```
AnonHugePages:         0 kB
ShmemHugePages:        0 kB
HugePages_Total:   30000
HugePages_Free:    20784
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
```
If `HugePages_Total` is equal to zero then hugh pages have not been configured correctly. If it is equal to 30000 then the hugh pages have been correctly configured

Hugh pages are not enabled in the [run_test_suite.sh](run_test_suite.sh) script.
