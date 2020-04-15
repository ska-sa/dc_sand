#ifndef MEM_RATE_TEST_H
#define MEM_RATE_TEST_H

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

/**\class MemRateTest
 * \brief   CUDA specific implementation of the PcieRateTest class
 * \details Implements all functions required by the PcieRateTest class for CUDA specific devices.
 */
class MemRateTest
{
    public:
        MemRateTest(int32_t i32NumThreads, int32_t i32BufferSize_bytes);

        ~MemRateTest();

        float transfer(int64_t i64NumTransfers);
        float transferForLenghtOfTime(int64_t i64NumSeconds);
        
    protected:
        ///Number of threads to execute in parallel
        int32_t m_i32NumThreads;

        ///Array of pointers to locations in memory to continuously read data from
        char ** m_ppMemArrays;

        ///Size of a single buffer
        int32_t m_i32BufferSize_bytes; 

        ///Device pointers
        //int8_t * m_pi32HInput;
        //int8_t * m_pi32HOutput; 

        ///Host pointers
        //int8_t * m_pi32DGpuArray;

        /// Stream for host to device data transfers
        //cudaStream_t m_streamH2D;
        /// Stream for device to host data transfers
        //cudaStream_t m_streamD2H;

        /// CUDA events for timing and synchronisation across treams
        //cudaEvent_t m_eventStart;
        //cudaEvent_t m_eventEnd;
        //cudaEvent_t m_pEventSync[NUM_SYNC_EVENTS];
};

#endif