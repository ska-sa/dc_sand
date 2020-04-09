#ifndef CUDA_PCIE_RATE_TEST_H
#define CUDA_PCIE_RATE_TEST_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

#include "pcieRateTest.h"

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


class CudaPcieRateTest : public PcieRateTest
{
    public:
        CudaPcieRateTest(int32_t i32DeviceId, int64_t i64NumFrames, int64_t i64FrameSizeBytes, int64_t i64NumTransfers ,bool bH2D, bool bD2H);
        ~CudaPcieRateTest();
        float transfer() override;
        static void list_gpus();

    protected:
        int8_t * m_pi32HInput;
        int8_t * m_pi32HOutput; 
        int8_t * m_pi32DGpuArray;

        cudaStream_t m_streamH2D;
        cudaStream_t m_streamD2H;
        cudaEvent_t m_eventStart;
        cudaEvent_t m_eventEnd;
        cudaEvent_t m_pEventSync[NUM_SYNC_EVENTS];
};

#endif