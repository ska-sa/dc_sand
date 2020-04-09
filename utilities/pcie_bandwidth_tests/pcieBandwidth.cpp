#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <boost/program_options.hpp>
#include <string>
#include <limits>

#include "cudaPcieRateTest.h"

#define NUM_TRANSFERS 5000
#define NUM_FRAMES 100
#define NUM_TESTS 3
#define FRAME_SIZE_BYTES 5000000

void list_gpus();

int main(int argc, char** argv){
    std::cout << "PCIe Bandwidth Tests" << std::endl;


    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("h2d,d", "Enable host to device stream")
        ("d2h,s", "Enable device to host stream")
        ("list_gpus,l", "List GPUs available on the current server")
        ("use_gpu_id,g", boost::program_options::value<int32_t>()->default_value(0) ,"Select GPU to use, use --list_gpus command to view GPU indexes")
        ("num_tests,t", boost::program_options::value<int64_t>()->default_value(NUM_TESTS), "Number of tests, set to negative for infinite")
        ("num_transfers_per_test,p", boost::program_options::value<int64_t>()->default_value(NUM_TRANSFERS), "Number of frames to transfer")
        ("frame_size,f", boost::program_options::value<int64_t>()->default_value(FRAME_SIZE_BYTES/1000000), "Frame size (MB)")
    ;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("list_gpus"))
    {
        list_gpus();
        return 1;
    }

    bool bH2D=false;
    bool bD2H=false;
    if (vm.count("h2d"))
    {
        bH2D=true;
        std::cout << "Enabled transfers from host to device" << "\n";
    }
    if (vm.count("d2h"))
    {
        bD2H=true;
        std::cout << "Enabled transfers from device to host" << "\n";
    }
    if(!bD2H && !bH2D)
    {
        std::cout << "ERROR: No PCIe transfer direction specified" << std::endl;
        return -1;
    }

    int64_t i64NumTests = vm["num_tests"].as<int64_t>();
    int64_t i64NumTransfers = vm["num_transfers_per_test"].as<int64_t>();
    int64_t i64FrameSize_bytes = vm["frame_size"].as<int64_t>()*1000*1000;
    int32_t gpuId = vm["use_gpu_id"].as<int32_t>();
    int i32DevicesCount;
    cudaGetDeviceCount(&i32DevicesCount);
    if(gpuId >= i32DevicesCount){
        std::cout << "ERROR: Invalid device index specified. Use --list_gpus flag to view valid indexes " << std::endl;
        return -1;
    }
    
    cudaPcieRateTest rateTest(gpuId,NUM_FRAMES,i64FrameSize_bytes,i64NumTransfers, bH2D, bD2H);

    if(i64NumTests < 1)
    {
        i64NumTests=std::numeric_limits<int64_t>::max();
        std::cout << "Running infinitly" << std::endl;
    }

    for (int64_t i64TestIndex = 0; i64TestIndex < i64NumTests; i64TestIndex++)
    {
        cudaPcieRateTest::TransferReturn retVal = rateTest.transfer();
        std::cout << "Test " << i64TestIndex+1 << " of " << i64NumTests << ": " << retVal.fDataRate_Gbps << " Gbps " << std::endl;
    }

    return 0;
}


void list_gpus(){
    int i32DevicesCount;
    cudaGetDeviceCount(&i32DevicesCount);
    std::cout << "Available Cuda Devices:" <<std::endl;
    for(int int32DeviceIndex = 0; int32DeviceIndex < i32DevicesCount; ++int32DeviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, int32DeviceIndex);
        std::cout << "\tIndex: " << int32DeviceIndex << " Device: " << deviceProperties.name << " PCIe Domain ID: " << deviceProperties.pciDomainID << std::endl;
    }
}

