#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <boost/program_options.hpp>
#include <string>
#include <limits>

#include "cudaPcieRateTest.hpp"

#define DEFAULT_NUM_TRANSFERS 5000
#define DEFAULT_NUM_FRAMES 100
#define DEFAULT_NUM_TESTS 3
#define DEFAULT_FRAME_SIZE_BYTES 5000000

int main(int argc, char** argv){
    std::cout << "PCIe Bandwidth Tests" << std::endl;


    /// Set up command line arguments
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("h2d,d", "Enable host to device stream")
        ("d2h,s", "Enable device to host stream")
        ("list_gpus,l", "List GPUs available on the current server")
        ("use_gpu_id,g", boost::program_options::value<int32_t>()->default_value(0) ,"Select GPU to use, use --list_gpus command to view GPU indexes")
        ("num_tests,t", boost::program_options::value<int64_t>()->default_value(DEFAULT_NUM_TESTS), "Number of tests, set to negative for infinite")
        ("num_transfers_per_test,p", boost::program_options::value<int64_t>()->default_value(DEFAULT_NUM_TRANSFERS), "Number of frames to transfer")
        ("frame_size,f", boost::program_options::value<int64_t>()->default_value(DEFAULT_FRAME_SIZE_BYTES/1000000), "Frame size (MB)")
    ;

    boost::program_options::variables_map clVariableMap;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), clVariableMap);
    boost::program_options::notify(clVariableMap);

    /// Retrieve and sanitise command line arguments
    if (clVariableMap.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    if (clVariableMap.count("list_gpus"))
    {
        CudaPcieRateTest::list_gpus();
        return 1;
    }

    bool bH2D=false;
    bool bD2H=false;
    if (clVariableMap.count("h2d"))
    {
        bH2D=true;
        std::cout << "Enabled transfers from host to device" << "\n";
    }
    if (clVariableMap.count("d2h"))
    {
        bD2H=true;
        std::cout << "Enabled transfers from device to host" << "\n";
    }
    if(!bD2H && !bH2D)
    {
        std::cout << "ERROR: No PCIe transfer direction specified" << std::endl;
        return -1;
    }

    int64_t i64NumTests = clVariableMap["num_tests"].as<int64_t>();
    if(i64NumTests < 1)
    {
        i64NumTests=std::numeric_limits<int64_t>::max();
        std::cout << "Running infinitly" << std::endl;
    }

    int64_t i64NumTransfers = clVariableMap["num_transfers_per_test"].as<int64_t>();
    int64_t i64FrameSize_bytes = clVariableMap["frame_size"].as<int64_t>()*1000*1000;
    int32_t gpuId = clVariableMap["use_gpu_id"].as<int32_t>();
    int i32DevicesCount;
    cudaGetDeviceCount(&i32DevicesCount);
    if(gpuId >= i32DevicesCount){
        std::cout << "ERROR: Invalid device index specified. Use --list_gpus flag to view valid indexes " << std::endl;
        return -1;
    }
    
    /// Initialise PCIe test
    CudaPcieRateTest cudaPcieRateTest(gpuId,DEFAULT_NUM_FRAMES,i64FrameSize_bytes, bH2D, bD2H);


    /// Run PCIe rate tests
    for (int64_t i64TestIndex = 0; i64TestIndex < i64NumTests; i64TestIndex++)
    {
        float fTransferRate_Gbps = cudaPcieRateTest.transfer(i64NumTransfers);
        std::cout << "Test " << i64TestIndex+1 << " of " << i64NumTests << ": " << fTransferRate_Gbps << " Gbps " << std::endl;
    }

    return 0;
}

