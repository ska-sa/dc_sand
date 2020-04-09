#include "cudaPcieRateTest.h"

cudaPcieRateTest::cudaPcieRateTest(int32_t i32GpuId, int64_t i64NumFrames, int64_t i64FrameSizeBytes, int64_t i64NumTransfers ,bool bH2D, bool bD2H):
    m_i32GpuId(i32GpuId),
    m_i64NumFrames(i64NumFrames),
    m_i64FrameSizeBytes(i64FrameSizeBytes),
    m_i64NumTransfers(i64NumTransfers),
    m_bH2D(bH2D),
    m_bD2H(bD2H)
{   
    m_i64ArraySize_bytes = m_i64NumFrames*m_i64FrameSizeBytes;
    gpuErrchk(cudaSetDevice(m_i32GpuId));
    gpuErrchk(cudaMalloc(&m_pi32DGpuArray, m_i64ArraySize_bytes));
    if(m_bH2D == 1)
    {
        gpuErrchk(cudaMallocHost(&m_pi32HInput, m_i64ArraySize_bytes));
        gpuErrchk(cudaStreamCreate(&m_streamH2D));
    }
    if(m_bD2H == 1)
    {
        gpuErrchk(cudaMallocHost(&m_pi32HOutput, m_i64ArraySize_bytes));
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

cudaPcieRateTest::~cudaPcieRateTest()
{
    gpuErrchk(cudaSetDevice(m_i32GpuId));
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

cudaPcieRateTest::TransferReturn cudaPcieRateTest::transfer(){
    gpuErrchk(cudaSetDevice(m_i32GpuId));
    if(m_bH2D == 1){
        gpuErrchk(cudaEventRecord(m_eventStart,m_streamH2D));
    }else{
        gpuErrchk(cudaEventRecord(m_eventStart,m_streamD2H));
    }

    for (size_t i = 0; i < m_i64NumTransfers; i++)
    {
        if(m_bH2D == 1){
            gpuErrchk(cudaMemcpyAsync(m_pi32DGpuArray+(i%m_i64NumFrames)*m_i64FrameSizeBytes,m_pi32HInput+(i%m_i64NumFrames)*m_i64FrameSizeBytes, m_i64FrameSizeBytes, cudaMemcpyHostToDevice, m_streamH2D));
            if(m_bD2H == 1)
            {
                gpuErrchk(cudaEventRecord(m_pEventSync[i % NUM_SYNC_EVENTS],m_streamH2D));
            }
        }

        if(m_bD2H == 1){
            if(m_bH2D == 1)
            {
                gpuErrchk(cudaStreamWaitEvent(m_streamD2H,m_pEventSync[i % NUM_SYNC_EVENTS],0));
            }
            gpuErrchk(cudaMemcpyAsync(m_pi32HOutput+(i%m_i64NumFrames)*m_i64FrameSizeBytes,m_pi32DGpuArray+(i%m_i64NumFrames)*m_i64FrameSizeBytes, m_i64FrameSizeBytes, cudaMemcpyDeviceToHost, m_streamD2H));
        }
        
        if((i%NUM_SYNC_EVENTS) == (NUM_SYNC_EVENTS-1)){
            cudaStreamSynchronize(m_streamD2H);
        }
    }

    if(m_bH2D == 1){
        gpuErrchk(cudaEventRecord(m_eventEnd,m_streamH2D));
    }else{
        gpuErrchk(cudaEventRecord(m_eventEnd,m_streamD2H));
    }
    gpuErrchk(cudaDeviceSynchronize());

    //============== Calculate and report on transfers ====
    float fElapsedTime_ms;
    float fTotalTransfer_bytes = (int64_t)m_i64FrameSizeBytes*m_i64NumTransfers;
    gpuErrchk(cudaEventElapsedTime(&fElapsedTime_ms,m_eventStart,m_eventEnd));
    return TransferReturn{fTotalTransfer_bytes/1000.0/1000.0/1000.0,fElapsedTime_ms/1000.0,(fTotalTransfer_bytes/1000.0/1000.0/1000.0*8.0)/(fElapsedTime_ms/1000.0)};
    //std::cout << "\tTotal Time: " << fElapsedTime_ms/1000.0 << " s" << std::endl; 
    //std::cout << "\tTransfer Size: " << fTotalTransfer_bytes/1000.0/1000.0/1000.0 << " Gb" << std::endl; 
    //std::cout << "\tData Rate: " << (fTotalTransfer_bytes/1000.0/1000.0/1000.0*8.0)/(fElapsedTime_ms/1000.0) << " Gbps" << std::endl; 

}