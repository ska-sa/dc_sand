#ifndef CUDA_PCIE_RATE_TEST_H
#define CUDA_PCIE_RATE_TEST_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

#include "pcieRateTest.hpp"

/// Defines number of events to use for synchronisation
#define NUM_SYNC_EVENTS 200

/// Standard CUDA error checking wrapper function
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/** \class   CudaPcieRateTest
 *  \brief   CUDA specific implementation of the PcieRateTest class
 *  \details Implements all functions required by the PcieRateTest class for CUDA specific devices.
 */
class CudaPcieRateTest : public PcieRateTest
{
    public:
        CudaPcieRateTest(int32_t i32DeviceId, int64_t i64NumFrames, int64_t i64FrameSizeBytes, bool bH2D, bool bD2H);

        ~CudaPcieRateTest();

        float transfer(int64_t i64NumTransfers) override;
        virtual float transferForLenghtOfTime(int64_t i64NumSeconds_s) override;
        
        /// Static function that returns a list of CUDA enabled GPUs as well as their device id for setting the correct value m_i32DeviceId. 
        static void list_gpus();

    protected:
        /// Host pointer to store data for host to device transfers
        int8_t * m_pi32HInput;

        /// Host pointer to store data for device to host transfers
        int8_t * m_pi32HOutput; 

        /// Device pointer to store data for both device to host and host to device transfers
        int8_t * m_pi32DGpuArray;

        /// Stream for host to device data transfers
        cudaStream_t m_streamH2D;
        /// Stream for device to host data transfers
        cudaStream_t m_streamD2H;

        /// CUDA events for timing and synchronisation across treams
        cudaEvent_t m_eventStart;
        cudaEvent_t m_eventEnd;
        cudaEvent_t m_pEventSync[NUM_SYNC_EVENTS];
};

#endif