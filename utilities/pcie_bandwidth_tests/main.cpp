#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <boost/program_options.hpp>
#include <string>
#include <limits>
#include <omp.h>

#include "cudaPcieRateTest.hpp"
#include "memRateTest.hpp"

#define DEFAULT_NUM_TRANSFERS 5000
#define DEFAULT_NUM_FRAMES 100
#define DEFAULT_NUM_TESTS 3
#define DEFAULT_FRAME_SIZE_BYTES 5000000

int main(int argc, char** argv){
    omp_set_nested(1);

    /// Set up command line arguments
    boost::program_options::options_description clDescription("Allowed options");
    clDescription.add_options()
        ("help,h", "Produce help message")
        ("mem_bw_test,b", "Perform memcpy test")
        ("h2d,d", "Enable host to device stream")
        ("d2h,s", "Enable device to host stream")
        ("csv,c", "Disables all human readable text and instead outputs the data in a csv format for easier exporting to external programs")
        ("list_gpus,l", "List GPUs available on the current server")
        ("gpu_id_mask,g", boost::program_options::value<int32_t>()->default_value(0) ,"Mask to set which GPUs to use e.g. 0101 will use GPU 0 and 2 while skipping 1 and 3")
        ("min_threads,m", boost::program_options::value<int64_t>()->default_value(0), "Minimum number of threads performing memcopies")
        ("max_threads,M", boost::program_options::value<int64_t>()->default_value(50), "Maximum number of threads performing memcopies")
        ("time_per_test_s,t", boost::program_options::value<int64_t>()->default_value(60), "Number of seconds to perform each test for")
        //("num_transfers_per_test,p", boost::program_options::value<int64_t>()->default_value(DEFAULT_NUM_TRANSFERS), "Number of frames to transfer")
        //("frame_size,f", boost::program_options::value<int64_t>()->default_value(DEFAULT_FRAME_SIZE_BYTES/1000000), "Frame size (MB)")
    ;

    boost::program_options::variables_map clVariableMap;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, clDescription), clVariableMap);
    boost::program_options::notify(clVariableMap);
    
    // Retrieve and sanitise command line arguments
    bool bCsvMode = false;
    if (clVariableMap.count("csv"))
    {
        bCsvMode=true;
    }

    if(!bCsvMode)
    {
        std::cout << "================================================================================" << std::endl;
        std::cout << "PCIe Bandwidth Tests" << std::endl;
        std::cout << std::endl;
    }
    
    if (clVariableMap.count("help"))
    {
        std::cout << clDescription << "\n";
        return 1;
    }

    if (clVariableMap.count("list_gpus"))
    {
        CudaPcieRateTest::list_gpus();
        std::cout << std::endl;
        return 1;
    }

    int64_t i64TestLength_s = clVariableMap["time_per_test_s"].as<int64_t>();
    if(i64TestLength_s<=0){
        std::cout << "ERROR: Time per test needs to be greater than 0" << std::endl;
        return -1;
    }
    int64_t i64MinThreads = clVariableMap["min_threads"].as<int64_t>();
    int64_t i64MaxThreads = clVariableMap["max_threads"].as<int64_t>();
    
    //Configure RAM bandwidth test
    bool bPerformMemBWTest = false;
    if (clVariableMap.count("mem_bw_test"))
    {
        bPerformMemBWTest = true;
        if(i64MinThreads>i64MaxThreads){
            std::cout << "ERROR: Minimum threads is greater than maximum threads" << std::endl;
            return -1;
        }
        if(i64MinThreads < 0 || i64MaxThreads < 0){
            std::cout << "ERROR: Number of threads needs to be <=0" << std::endl;
            return -1;
        }
        if(!bCsvMode)
        {
            std::cout << "Testing System RAM Bandwidth with a thread count ranging from " << i64MinThreads << " to " << i64MaxThreads << std::endl;
        }
    }else{
        i64MinThreads = 1;
        i64MaxThreads = 1;
    }
    omp_set_num_threads(i64MaxThreads + 20);

    //Configure PCIe bandwidth Tests
    int i32DevicesCount;
    cudaGetDeviceCount(&i32DevicesCount);
    bool bH2D=false;
    bool bD2H=false;
    bool pbGPUMask[i32DevicesCount];
    int32_t pbUseGPU = clVariableMap["gpu_id_mask"].as<int32_t>();
    if(pbUseGPU != 0){
        if (clVariableMap.count("h2d"))
        {
            bH2D=true;
            if(!bCsvMode)
            {
                std::cout << "Enabled transfers from host to device" << "\n";
            }
        }
        if (clVariableMap.count("d2h"))
        {
            bD2H=true;
            if(!bCsvMode)
            {
                std::cout << "Enabled transfers from device to host" << "\n";
            }
        }
        if(!bD2H && !bH2D)
        {
            std::cout << "ERROR: No PCIe transfer direction specified" << std::endl;
            return -1;
        }
        for (int32_t i32DeviceId = 0; i32DeviceId < i32DevicesCount; i32DeviceId++)
        {
            //std::cout << 
            uint8_t u8BitSlice = pbUseGPU%10;
            pbUseGPU = pbUseGPU/10;
            if(u8BitSlice == 1){
                pbGPUMask[i32DeviceId] = true;
                if(!bCsvMode)
                {   
                    std::cout << "Using GPU Index: " << i32DeviceId << std::endl;
                }
            }
        }
        pbUseGPU = 1;
        if(!bCsvMode)
        {
            std::cout << "Allocated " << DEFAULT_FRAME_SIZE_BYTES*DEFAULT_NUM_FRAMES/1000.0/1000.0/1000.0 << " GB of device memory per GPU." << std::endl;
        }
    }

    if(pbUseGPU==0 && bPerformMemBWTest==false){
        std::cout << "ERROR: Need to use at least -m or -g 1 flags. At the moment no tests are specified. " << std::endl;
        return -1;
    }
        
    
    //Begin bandwidth tests
    if(!bCsvMode)
    {
        //Prints out column titles
        std::cout << std::setw(20) << "No. of Threads" << std::setprecision(2) 
            << std::setw(20) << "Mem BW GBps";
        for (int32_t i32DeviceId = 0; i32DeviceId < i32DevicesCount; i32DeviceId++)
        {
            if(pbGPUMask[i32DeviceId] == true)
            {
                std::cout << std::setw(11) << "GPU " << std::setw(1) << std::setprecision(1) << i32DeviceId << " PCI BW(Gbps)";    
            }
        }
        std::cout << std::endl; 
    }

    //Iterates through bandwidth tests incrementing the number of threads performing ram copies. i32Thread
    for (int32_t i32ThreadCount = i64MinThreads; i32ThreadCount < i64MaxThreads+1; i32ThreadCount++)
    {
        float fMemRate_GBps;
        float fPcieRate_Gbps[i32DevicesCount];
        
        //Two different sections to run in parallel - one performing bandwidth tests the other performing PCIe tests. Each section creates its own threads as necessary
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                //Performs memory bandwidth tests
                if(bPerformMemBWTest)
                {
                    MemRateTest memRateTest(i32ThreadCount,256*2048*2048);    
                    fMemRate_GBps = memRateTest.transferForLenghtOfTime(i64TestLength_s);
                }
            }
            
            #pragma omp section
            {
                //Performs PCIe bandwidth tests. Creates a seperate thread for each PCIe device being tested
                #pragma omp parallel for
                for (int32_t i32DeviceId = 0; i32DeviceId < i32DevicesCount; i32DeviceId++)
                {       
                    if(pbGPUMask[i32DeviceId] == true)
                    {
                        CudaPcieRateTest cudaPcieRateTest(i32DeviceId,DEFAULT_NUM_FRAMES,DEFAULT_FRAME_SIZE_BYTES, bH2D, bD2H);
                        fPcieRate_Gbps[i32DeviceId] = cudaPcieRateTest.transferForLenghtOfTime(i64TestLength_s);
                    }
                }
            }    
        }

        //Prints out number of threads and RAM bandwidth
        if(!bCsvMode)
        {
            std::cout << std::setw(20) << i32ThreadCount << std::setprecision(2) 
                << std::setw(20) << std::setprecision(6) << fMemRate_GBps;
        }else{
            std::cout << i32ThreadCount << "," << fMemRate_GBps;
        }
        
        //Prints out PCIe bandwidth
        for (int32_t i32DeviceId = 0; i32DeviceId < i32DevicesCount; i32DeviceId++)
        {
            if(pbGPUMask[i32DeviceId] == true)
            {
                if(!bCsvMode)
                {
                    std::cout << std::setw(25) << std::setprecision(6) << fPcieRate_Gbps[i32DeviceId];
                }else{
                    std::cout << "," << fPcieRate_Gbps[i32DeviceId];
                }
            }
        }
        std::cout << std::endl; 
    }
    
    if(!bCsvMode)
    {
        std::cout << std::endl;
        std::cout << "Done" << std::endl;
        std::cout << "================================================================================" << std::endl;
    }
    return 0;
}

