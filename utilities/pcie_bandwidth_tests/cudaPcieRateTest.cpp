#include "cudaPcieRateTest.hpp"

CudaPcieRateTest::CudaPcieRateTest(int32_t i32DeviceId, int64_t i64NumFrames, int64_t i64FrameSizeBytes ,bool bH2D, bool bD2H):
    PcieRateTest(i32DeviceId ,i64NumFrames ,i64FrameSizeBytes, bH2D, bD2H)
{   
    m_i64DeviceBufferSize_bytes = m_i64NumFrames*m_i64FrameSizeBytes;

    /// Allocates device and host CUDA buffers, creates all required streams and syncrhonisation events
    gpuErrchk(cudaSetDevice(m_i32DeviceId));
    gpuErrchk(cudaMalloc(&m_pi32DGpuArray, m_i64DeviceBufferSize_bytes));
    if(m_bH2D == 1)
    {
        gpuErrchk(cudaMallocHost(&m_pi32HInput, m_i64DeviceBufferSize_bytes));
        gpuErrchk(cudaStreamCreate(&m_streamH2D));
    }
    if(m_bD2H == 1)
    {
        gpuErrchk(cudaMallocHost(&m_pi32HOutput, m_i64DeviceBufferSize_bytes));
        gpuErrchk(cudaStreamCreate(&m_streamD2H));
    }

    if(m_bD2H == 1 && m_bH2D == 1)
    {
        for (size_t i = 0; i < NUM_SYNC_EVENTS; i++)
        {
            gpuErrchk(cudaEventCreate(&m_pEventSync[i]));
        }
        
    }

    gpuErrchk(cudaEventCreate(&m_eventStart));
    gpuErrchk(cudaEventCreate(&m_eventEnd));
}

CudaPcieRateTest::~CudaPcieRateTest()
{
    /// Destroys all events and streams. Frees all allocated buffers
    gpuErrchk(cudaSetDevice(m_i32DeviceId));
    gpuErrchk(cudaFree(m_pi32DGpuArray));
    if(m_bH2D == 1)
    {
        gpuErrchk(cudaFreeHost(m_pi32HInput));
        gpuErrchk(cudaStreamDestroy(m_streamH2D));
    }

    if(m_bD2H == 1)
    {
        gpuErrchk(cudaFreeHost(m_pi32HOutput));
        gpuErrchk(cudaStreamDestroy(m_streamD2H));
    }

    gpuErrchk(cudaEventDestroy(m_eventStart));
    gpuErrchk(cudaEventDestroy(m_eventEnd));
    if(m_bD2H == 1 && m_bH2D == 1)
    {
        for (size_t i = 0; i < NUM_SYNC_EVENTS; i++)
        {
            gpuErrchk(cudaEventDestroy(m_pEventSync[i]));
        }
    }
}

float CudaPcieRateTest::transfer(int64_t i64NumTransfers){
    /// Put timing start event on requried stream
    gpuErrchk(cudaSetDevice(m_i32DeviceId));
    if(m_bH2D == 1){
        gpuErrchk(cudaEventRecord(m_eventStart,m_streamH2D));
    }else{
        gpuErrchk(cudaEventRecord(m_eventStart,m_streamD2H));
    }

    /// Transfer data between host and device
    for (size_t i = 0; i < i64NumTransfers; i++)
    {
        if(m_bH2D == 1){
            gpuErrchk(cudaMemcpyAsync(m_pi32DGpuArray+(i%m_i64NumFrames)*m_i64FrameSizeBytes,m_pi32HInput+(i%m_i64NumFrames)*m_i64FrameSizeBytes, m_i64FrameSizeBytes, cudaMemcpyHostToDevice, m_streamH2D));
            if(m_bD2H == 1)
            {   
                /// Record memcpy to device as complete
                gpuErrchk(cudaEventRecord(m_pEventSync[i % NUM_SYNC_EVENTS],m_streamH2D));
            }
        }

        if(m_bD2H == 1){
            if(m_bH2D == 1)
            {
                /// Wait until memcpy to device of a specific frame is complete before transferring it back to the host.
                gpuErrchk(cudaStreamWaitEvent(m_streamD2H,m_pEventSync[i % NUM_SYNC_EVENTS],0));
            }
            gpuErrchk(cudaMemcpyAsync(m_pi32HOutput+(i%m_i64NumFrames)*m_i64FrameSizeBytes,m_pi32DGpuArray+(i%m_i64NumFrames)*m_i64FrameSizeBytes, m_i64FrameSizeBytes, cudaMemcpyDeviceToHost, m_streamD2H));
        }
        
        if((i%NUM_SYNC_EVENTS) == (NUM_SYNC_EVENTS-1)){
            cudaStreamSynchronize(m_streamD2H);
        }
    }

    /// Put timing stop event on requried stream
    if(m_bH2D == 1){
        gpuErrchk(cudaEventRecord(m_eventEnd,m_streamH2D));
    }else{
        gpuErrchk(cudaEventRecord(m_eventEnd,m_streamD2H));
    }

    /// Wait for all streams to finish processing
    gpuErrchk(cudaDeviceSynchronize());

    /// Calculate and return data rate
    float fElapsedTime_ms;
    float fTotalTransfer_bytes = (int64_t)m_i64FrameSizeBytes*i64NumTransfers;
    gpuErrchk(cudaEventElapsedTime(&fElapsedTime_ms,m_eventStart,m_eventEnd));

    float fTotalTransfer_Gbytes = fTotalTransfer_bytes/1000.0/1000.0/1000.0;
    float fTransferTime_s = fElapsedTime_ms/1000.0;
    float fTransferRate_Gbps = fTotalTransfer_Gbytes/fTransferTime_s*8.0;
    return fTransferRate_Gbps;
}

void CudaPcieRateTest::list_gpus()
{
    int i32DevicesCount;
    cudaGetDeviceCount(&i32DevicesCount);
    std::cout << "Available Cuda Devices:" <<std::endl;
    for(int int32DeviceIndex = 0; int32DeviceIndex < i32DevicesCount; ++int32DeviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, int32DeviceIndex);
        std::cout << "\tDevice ID: " << int32DeviceIndex << ". Device: " << deviceProperties.name << " PCIe Domain ID: " << deviceProperties.pciDomainID << std::endl;
    }
}