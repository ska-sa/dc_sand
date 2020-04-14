#include "openclPcieRateTest.hpp"

OpenCLPcieRateTest::OpenCLPcieRateTest(int32_t i32DeviceId, int64_t i64NumFrames, int64_t i64FrameSizeBytes ,bool bH2D, bool bD2H):
    PcieRateTest(i32DeviceId ,i64NumFrames ,i64FrameSizeBytes, bH2D, bD2H)
{   
    m_i64DeviceBufferSize_bytes = m_i64NumFrames*m_i64FrameSizeBytes;

    //Initialise the OpenCL device corrrectly
    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    if (allPlatforms.size() == 0)
    {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    // Initialise OpenCL device
    cl::Platform openClPlatform = allPlatforms[get_opencl_platform_id_from_index(i32DeviceId)];
    std::vector<cl::Device> openClDevices;
    openClPlatform.getDevices(CL_DEVICE_TYPE_ALL, &openClDevices); 
    cl::Device openClDevice = openClDevices[get_opencl_device_id_from_index(i32DeviceId)];
    context = cl::Context({openClDevice});
    std::cout << "Device: " << openClDevice.getInfo<CL_DEVICE_NAME>() << std::endl;

    /// Allocates device and host OpenCL buffers, creates all required streams and syncrhonisation events
    cl_int cliError = CL_MAP_FAILURE;
    m_pi32DGpuArray = cl::Buffer(context, CL_MEM_READ_WRITE, m_i64DeviceBufferSize_bytes, NULL, &cliError);
    if (cliError != CL_SUCCESS)
    {
        std::cout << "Error creating OpenCL m_pi32DGpuArray buffer" << cliError << std::endl;
    }

    void *pTempPtr = nullptr;
    if(m_bH2D == 1)
    {
            // Create input array on the host and then assign it to an OpenCL buffer
            if (posix_memalign(&pTempPtr, 4096, m_i64DeviceBufferSize_bytes)){
                std::cout << "Failed to create a m_pi32HInput buffer " << std::endl;
            }
            m_pi32HInput = reinterpret_cast<int8_t *>(pTempPtr);
            m_pi32HInputClBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , m_i64DeviceBufferSize_bytes, m_pi32HInput, &cliError);
            if (cliError != CL_SUCCESS)
            {
                std::cout << "Error creating OpenCL m_pi32HInputClBuffer buffer" << cliError << std::endl;
            }
            m_queueH2D = cl::CommandQueue(context, CL_QUEUE_PROFILING_ENABLE);
            
    }
    if(m_bD2H == 1)
    {
            // Create output array on the host and then assign it to an OpenCL buffer
            if (posix_memalign(&pTempPtr, 4096, m_i64DeviceBufferSize_bytes)){
                std::cout << "Failed to create a m_pi32HOutput buffer " << std::endl;
            }
            m_pi32HOutput = reinterpret_cast<int8_t *>(pTempPtr);
            m_pi32HOutputClBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , m_i64DeviceBufferSize_bytes, m_pi32HOutput, &cliError);
            if (cliError != CL_SUCCESS)
            {
                std::cout << "Error creating OpenCL m_pi32HInputClBuffer buffer" << cliError << std::endl;
            }
            m_queueD2H = cl::CommandQueue(context, CL_QUEUE_PROFILING_ENABLE);
    }

    // if(m_bD2H == 1 && m_bH2D == 1)
    // {
    //     for (size_t i = 0; i < NUM_SYNC_EVENTS; i++)
    //     {
    //         gpuErrchk(cudaEventCreate(&m_pEventSync[i]));
    //     }
        
    // }

    // gpuErrchk(cudaEventCreate(&m_eventStart));
    // gpuErrchk(cudaEventCreate(&m_eventEnd));

    std::cout << "asd1" << std::endl;
}

OpenCLPcieRateTest::~OpenCLPcieRateTest()
{
    /// Destroys all events and streams. Farees all allocated buffers
    //gpuErrchk(cudaSetDevice(m_i32DeviceId));
    //gpuErrchk(cudaFree(m_pi32DGpuArray));
    if(m_bH2D == 1)
    {
        free(m_pi32HInput);
        //gpuErrchk(cudaStreamDestroy(m_streamH2D));
    }

    if(m_bD2H == 1)
    {
        free(m_pi32HOutput);
        //gpuErrchk(cudaStreamDestroy(m_streamD2H));
    }

    //gpuErrchk(cudaEventDestroy(m_eventStart));
    //gpuErrchk(cudaEventDestroy(m_eventEnd));
    if(m_bD2H == 1 && m_bH2D == 1)
    {
        for (size_t i = 0; i < NUM_SYNC_EVENTS; i++)
        {
            //gpuErrchk(cudaEventDestroy(m_pEventSync[i]));
        }
    }
    std::cout << "das" << std::endl;
}

float OpenCLPcieRateTest::transfer(int64_t i64NumTransfers){
    /// Put timing start event on requried stream
    //gpuErrchk(cudaSetDevice(m_i32DeviceId));
    if(m_bH2D == 1){
        //gpuErrchk(cudaEventRecord(m_eventStart,m_streamH2D));
    }else{
        //gpuErrchk(cudaEventRecord(m_eventStart,m_streamD2H));
    }

    /// Transfer data between host and device
    for (size_t i = 0; i < 1/*i64NumTransfers*/; i++)
    {
        if(m_bH2D == 1){
            m_queueH2D.enqueueWriteBuffer(m_pi32HInputClBuffer,CL_TRUE, 0, m_i64FrameSizeBytes, m_pi32HInput, NULL, NULL); //.enqueueWriteBuffer(m_pi32HInputClBuffer,CL_FALSE,0,m_i64FrameSizeBytes,)
            //gpuErrchk(cudaMemcpyAsync(m_pi32DGpuArray+(i%m_i64NumFrames)*m_i64FrameSizeBytes,m_pi32HInput+(i%m_i64NumFrames)*m_i64FrameSizeBytes, m_i64FrameSizeBytes, cudaMemcpyHostToDevice, m_streamH2D));
            if(m_bD2H == 1)
            {   
                /// Record memcpy to device as complete
                //gpuErrchk(cudaEventRecord(m_pEventSync[i % NUM_SYNC_EVENTS],m_streamH2D));
            }
        }

        if(m_bD2H == 1){
            if(m_bH2D == 1)
            {
                /// Wait until memcpy to device of a specific frame is complete before transferring it back to the host.
                //gpuErrchk(cudaStreamWaitEvent(m_streamD2H,m_pEventSync[i % NUM_SYNC_EVENTS],0));
            }
            //gpuErrchk(cudaMemcpyAsync(m_pi32HOutput+(i%m_i64NumFrames)*m_i64FrameSizeBytes,m_pi32DGpuArray+(i%m_i64NumFrames)*m_i64FrameSizeBytes, m_i64FrameSizeBytes, cudaMemcpyDeviceToHost, m_streamD2H));
        }
        
        if((i%NUM_SYNC_EVENTS) == (NUM_SYNC_EVENTS-1)){
            //cudaStreamSynchronize(m_streamD2H);
        }
    }

    /// Put timing stop event on requried stream
    if(m_bH2D == 1){
        //gpuErrchk(cudaEventRecord(m_eventEnd,m_streamH2D));
    }else{
        //gpuErrchk(cudaEventRecord(m_eventEnd,m_streamD2H));
    }

    /// Wait for all streams to finish processing
    //gpuErrchk(cudaDeviceSynchronize());

    /// Calculate and return data rate
    float fElapsedTime_ms;
    float fTotalTransfer_bytes = (int64_t)m_i64FrameSizeBytes*i64NumTransfers;
    //gpuErrchk(cudaEventElapsedTime(&fElapsedTime_ms,m_eventStart,m_eventEnd));

    float fTotalTransfer_Gbytes = fTotalTransfer_bytes/1000.0/1000.0/1000.0;
    float fTransferTime_s = fElapsedTime_ms/1000.0;
    float fTransferRate_Gbps = fTotalTransfer_Gbytes/fTransferTime_s*8.0;
    return fTransferRate_Gbps;
}

void OpenCLPcieRateTest::list_opencl_devices()
{
    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    if (allPlatforms.size() == 0)
    {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    //std::cout << all_platforms.size() << " platforms found.\n";
    int32_t i32PlatformId = 0;
    int32_t i32Index = 0;

    std::cout << "Available OpenCL Devices:" <<std::endl;
    for (cl::Platform platform : allPlatforms)
    {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
        int i32DeviceId = 0;
        for (cl::Device device : devices)
        {
            //std::cout << "\tDevice ID" << platform_id++ << ": Device Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cout << "\tDevice ID: " << i32Index++ << ". Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        }
        i32PlatformId++;
    }
    std::cout << std::endl;
}

int32_t OpenCLPcieRateTest::get_opencl_device_id_from_index(int32_t i32Index){
    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    if (allPlatforms.size() == 0)
    {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    int32_t i32PlatformId = 0;
    int32_t i32IndexCurrent = 0;

    for (cl::Platform platform : allPlatforms)
    {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
        int i32DeviceId = 0;
        for (cl::Device device : devices)
        {
            if(i32IndexCurrent==i32Index){
                return i32DeviceId;
            }
            i32DeviceId++;
            i32IndexCurrent++;
        }
        i32PlatformId++;
    }
    std::cout << std::endl;
}

int32_t OpenCLPcieRateTest::get_opencl_platform_id_from_index(int32_t i32Index){
    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);
    if (allPlatforms.size() == 0)
    {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    int32_t i32PlatformId = 0;
    int32_t i32IndexCurrent = 0;

    for (cl::Platform platform : allPlatforms)
    {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
        int i32DeviceId = 0;
        for (cl::Device device : devices)
        {
            if(i32IndexCurrent==i32Index){
                return i32PlatformId;
            }
            i32IndexCurrent++;
            i32DeviceId++;
        }
        i32PlatformId++;
    }
    std::cout << std::endl;
}