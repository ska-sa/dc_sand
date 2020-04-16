#ifndef CUDA_PCIE_RATE_TEST_H
#define CUDA_PCIE_RATE_TEST_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <iostream>
#include <cstring>
#include "../../common/Utils.hpp"
#include "pcieRateTest.hpp"

/// Defines number of events to use for synchronisation
#define NUM_SYNC_EVENTS 200

/** \class   CudaPcieRateTest
 *  \brief   CUDA-specific implementation of the PcieRateTest class
 *  \details Implements all functions required by the PcieRateTest class for CUDA specific devices.
 */
class CudaPcieRateTest : public PcieRateTest
{
    public:
        CudaPcieRateTest(int32_t i32DeviceId, size_t ulNumFrames, size_t ulFrameSizeBytes ,bool bH2D, bool bD2H);

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

        /// CUDA event for logging the starting time of a data stream
        cudaEvent_t m_eventStart;
        /// CUDA event for logging the ending time of a data stream
        cudaEvent_t m_eventEnd;
        /// CUDA events for syncing the H2D and D2H transfers so that the same frame is not transmitted and received simultaneously
        cudaEvent_t m_pEventSync[NUM_SYNC_EVENTS];
};

#endif