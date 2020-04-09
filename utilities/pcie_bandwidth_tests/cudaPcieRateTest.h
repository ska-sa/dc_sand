#ifndef CUDA_PCIE_RATE_TEST_H
#define CUDA_PCIE_RATE_TEST_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

#define NUM_SYNC_EVENTS 100

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class cudaPcieRateTest
{
private:
    int32_t m_i32GpuId;

    int64_t m_i64NumFrames;
    int64_t m_i64FrameSizeBytes;
    int64_t m_i64NumTransfers;
    int64_t m_i64ArraySize_bytes;

    bool m_bH2D;
    bool m_bD2H;

    int8_t * m_pi32HInput;
    int8_t * m_pi32HOutput; 
    int8_t * m_pi32DGpuArray;

    cudaStream_t m_streamH2D;
    cudaStream_t m_streamD2H;
    cudaEvent_t m_eventStart;
    cudaEvent_t m_eventEnd;
    cudaEvent_t m_pEventSync[NUM_SYNC_EVENTS];
    
public:
    struct TransferReturn {
        float fTransferSize_Gb;
        float fTransferTime_s;
        float fDataRate_Gbps;
    };
    cudaPcieRateTest(int32_t i32GpuId, int64_t i64NumFrames, int64_t i64FrameSizeBytes, int64_t i64NumTransfers ,bool bH2D, bool bD2H);
    ~cudaPcieRateTest();
    cudaPcieRateTest::TransferReturn transfer();
};

#endif