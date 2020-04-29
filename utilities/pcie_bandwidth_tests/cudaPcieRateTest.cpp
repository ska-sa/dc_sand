#include "cudaPcieRateTest.hpp"
#include <chrono>
#include <iostream>

CudaPcieRateTest::CudaPcieRateTest(int32_t i32DeviceId, size_t ulNumFrames, size_t ulFrameSizeBytes, bool bH2D, bool bD2H):
    PcieRateTest(i32DeviceId, ulNumFrames, ulFrameSizeBytes, bH2D, bD2H)
{   
    // Allocates device and host CUDA buffers, creates all required streams and syncrhonisation events
    GPU_ERRCHK(cudaSetDevice(m_i32DeviceId));
    GPU_ERRCHK(cudaMalloc(&m_pi32DGpuArray, m_ulDeviceBufferSize_bytes));
    if(m_bH2D == 1)
    {
        GPU_ERRCHK(cudaMallocHost(&m_pi32HInput, m_ulDeviceBufferSize_bytes));
        GPU_ERRCHK(cudaStreamCreate(&m_streamH2D));
    }
    if(m_bD2H == 1)
    {
        GPU_ERRCHK(cudaMallocHost(&m_pi32HOutput, m_ulDeviceBufferSize_bytes));
        GPU_ERRCHK(cudaStreamCreate(&m_streamD2H));
    }

    if(m_bD2H == 1 && m_bH2D == 1)
    {
        for (size_t i = 0; i < NUM_SYNC_EVENTS; i++)
        {
            GPU_ERRCHK(cudaEventCreate(&m_pEventSync[i]));
        }
        
    }

    GPU_ERRCHK(cudaEventCreate(&m_eventStart));
    GPU_ERRCHK(cudaEventCreate(&m_eventEnd));
}

CudaPcieRateTest::~CudaPcieRateTest()
{
    // Destroys all events and streams. Frees all allocated buffers
    GPU_ERRCHK(cudaSetDevice(m_i32DeviceId));
    GPU_ERRCHK(cudaFree(m_pi32DGpuArray));
    if(m_bH2D == 1)
    {
        GPU_ERRCHK(cudaFreeHost(m_pi32HInput));
        GPU_ERRCHK(cudaStreamDestroy(m_streamH2D));
    }

    if(m_bD2H == 1)
    {
        GPU_ERRCHK(cudaFreeHost(m_pi32HOutput));
        GPU_ERRCHK(cudaStreamDestroy(m_streamD2H));
    }

    GPU_ERRCHK(cudaEventDestroy(m_eventStart));
    GPU_ERRCHK(cudaEventDestroy(m_eventEnd));
    if(m_bD2H == 1 && m_bH2D == 1)
    {
        for (size_t i = 0; i < NUM_SYNC_EVENTS; i++)
        {
            GPU_ERRCHK(cudaEventDestroy(m_pEventSync[i]));
        }
    }
}

float CudaPcieRateTest::transfer(int64_t i64NumTransfers){
    // Put timing start event on required stream
    GPU_ERRCHK(cudaSetDevice(m_i32DeviceId));
    if(m_bH2D == 1){
        GPU_ERRCHK(cudaEventRecord(m_eventStart,m_streamH2D));
    }else{
        GPU_ERRCHK(cudaEventRecord(m_eventStart,m_streamD2H));
    }

    // Transfer data between host and device
    for (size_t i = 0; i < i64NumTransfers; i++)
    {
        if(m_bH2D == 1){
            GPU_ERRCHK(cudaMemcpyAsync(m_pi32DGpuArray + (i % m_ulNumFrames)*m_ulFrameSizeBytes,m_pi32HInput + (i % m_ulNumFrames)*m_ulFrameSizeBytes, m_ulFrameSizeBytes, cudaMemcpyHostToDevice, m_streamH2D));
            if(m_bD2H == 1)
            {   
                // Record memcpy to device as complete
                GPU_ERRCHK(cudaEventRecord(m_pEventSync[i % NUM_SYNC_EVENTS],m_streamH2D));
            }
        }

        if(m_bD2H == 1){
            if(m_bH2D == 1)
            {
                // Wait until memcpy to device of a specific frame is complete before transferring it back to the host.
                GPU_ERRCHK(cudaStreamWaitEvent(m_streamD2H,m_pEventSync[i % NUM_SYNC_EVENTS],0));
            }
            GPU_ERRCHK(cudaMemcpyAsync(m_pi32HOutput + (i % m_ulNumFrames)*m_ulFrameSizeBytes,m_pi32DGpuArray + (i % m_ulNumFrames)*m_ulFrameSizeBytes, m_ulFrameSizeBytes, cudaMemcpyDeviceToHost, m_streamD2H));
        }
        
        // Wait until all buffers have finished processing. 
        // There may be a more efficient way to do this by waiting on individual events instead of an entire stream but this will maybe gain 1% or 2% more performance.
        if((i % NUM_SYNC_EVENTS) == (NUM_SYNC_EVENTS - 1)){
            if(m_bD2H == 1){
                GPU_ERRCHK(cudaStreamSynchronize(m_streamD2H));
            }else{
                GPU_ERRCHK(cudaStreamSynchronize(m_streamH2D));
            }
        }
    }

    // Put timing stop event on requried stream
    if(m_bH2D == 1){
        GPU_ERRCHK(cudaEventRecord(m_eventEnd,m_streamH2D));
    }else{
        GPU_ERRCHK(cudaEventRecord(m_eventEnd,m_streamD2H));
    }

    // Wait for all streams to finish processing
    GPU_ERRCHK(cudaDeviceSynchronize());

    // Calculate and return data rate
    float fElapsedTime_ms;
    float fTotalTransfer_bytes = (int64_t)m_ulFrameSizeBytes*i64NumTransfers;
    GPU_ERRCHK(cudaEventElapsedTime(&fElapsedTime_ms,m_eventStart,m_eventEnd));

    float fTotalTransfer_Gbytes = fTotalTransfer_bytes/1000.0/1000.0/1000.0;
    float fTransferTime_s = fElapsedTime_ms/1000.0;
    float fTransferRate_Gbps = fTotalTransfer_Gbytes/fTransferTime_s*8.0;
    return fTransferRate_Gbps;
}

float CudaPcieRateTest::transferForLengthOfTime(int64_t i64NumSeconds_s){
    // Put timing start event on required stream
    GPU_ERRCHK(cudaSetDevice(m_i32DeviceId));
    if(m_bH2D == 1){
        GPU_ERRCHK(cudaEventRecord(m_eventStart,m_streamH2D));
    }else{
        GPU_ERRCHK(cudaEventRecord(m_eventStart,m_streamD2H));
    }

    int64_t i64NumTransfers = 0;
    std::chrono::duration<double> timeElapsed_s;
    auto start = std::chrono::high_resolution_clock::now();
    // Transfer data between host and device. This transfer will continue until i64NumSeconds_s have passed
    do
    {
        int64_t i = i64NumTransfers;
        if(m_bH2D == 1){
            GPU_ERRCHK(cudaMemcpyAsync(m_pi32DGpuArray+(i%m_ulNumFrames)*m_ulFrameSizeBytes,m_pi32HInput+(i%m_ulNumFrames)*m_ulFrameSizeBytes, m_ulFrameSizeBytes, cudaMemcpyHostToDevice, m_streamH2D));
            if(m_bD2H == 1)
            {   
                // Record memcpy to device as complete
                GPU_ERRCHK(cudaEventRecord(m_pEventSync[i % NUM_SYNC_EVENTS],m_streamH2D));
            }
        }

        if(m_bD2H == 1){
            if(m_bH2D == 1)
            {
                // Wait until memcpy to device of a specific frame is complete before transferring it back to the host.
                GPU_ERRCHK(cudaStreamWaitEvent(m_streamD2H,m_pEventSync[i % NUM_SYNC_EVENTS],0));
            }
            GPU_ERRCHK(cudaMemcpyAsync(m_pi32HOutput+(i%m_ulNumFrames)*m_ulFrameSizeBytes,m_pi32DGpuArray+(i%m_ulNumFrames)*m_ulFrameSizeBytes, m_ulFrameSizeBytes, cudaMemcpyDeviceToHost, m_streamD2H));
        }
        
        // Wait until all buffers have finished processing. 
        // There may be a more efficient way to do this by waiting on individual events instead of an entire stream but this will maybe gain 1% or 2% more performance.
        if((i%NUM_SYNC_EVENTS) == (NUM_SYNC_EVENTS-1)){
            if(m_bD2H == 1){
                GPU_ERRCHK(cudaStreamSynchronize(m_streamD2H));
            }else{
                GPU_ERRCHK(cudaStreamSynchronize(m_streamH2D));
            }
        }
        i64NumTransfers++;
        auto now = std::chrono::high_resolution_clock::now();
        timeElapsed_s = now - start;
    } while (timeElapsed_s.count() < i64NumSeconds_s);

    // Put timing stop event on requried stream
    if(m_bH2D == 1){
        GPU_ERRCHK(cudaEventRecord(m_eventEnd,m_streamH2D));
    }else{
        GPU_ERRCHK(cudaEventRecord(m_eventEnd,m_streamD2H));
    }

    // Wait for all streams to finish processing
    GPU_ERRCHK(cudaDeviceSynchronize());

    // Calculate and return data rate
    float fElapsedTime_ms;
    float fTotalTransfer_bytes = (int64_t)m_ulFrameSizeBytes*i64NumTransfers;
    GPU_ERRCHK(cudaEventElapsedTime(&fElapsedTime_ms,m_eventStart,m_eventEnd));

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
    for(int i32DeviceIndex = 0; i32DeviceIndex < i32DevicesCount; ++i32DeviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, i32DeviceIndex);
        std::cout << "\tDevice ID: " << i32DeviceIndex << ". Device: " << deviceProperties.name << " PCIe Domain ID: " << deviceProperties.pciDomainID << std::endl;
    }
}