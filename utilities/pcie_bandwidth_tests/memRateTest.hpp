#ifndef MEM_RATE_TEST_H
#define MEM_RATE_TEST_H

#include <stdint.h>
#include <stdio.h>

/** \class      MemRateTest
 *  \brief      Measure the system RAM bandwidth
 *  \details    The MemRateTest class measures the system RAM bandwidth. To perform the transfers, a
 *              function was written in assembly to read large chunks of data from system RAM into 256-bit
 *              wide AVX registers on the CPU. The assembly functions do not get optomised by the compiler
 *              - this is desirable in this instance. This class provides the functionality for spawning 
 *              multiple threads to perform transfers as more threads generally results in increased memory
 *              bandwidth until memory bus saturation.
 */
class MemRateTest
{
    public:
        
        /// The default constructor is disabled.
        MemRateTest() = delete;

        /** Constructs the MemRateTest class. Sets the nmber of threads that must transfer as well as 
         *  the amount of memory to be allocated by a single thread.
         *  \param i32NumThreads Specify the number of threads to transfer in parallel.
         *  \param i32BufferSize_bytes Specify the size of the buffer in bytes to allocate per thread.
         *  \param useHughPages Set true to use hugh pages. Will throw an error if OS is not configured properly
         */
        MemRateTest(size_t ulNumThreads, size_t ulBufferSize_bytes, bool useHugePages);

        /// Destructor releases all assigned buffers.
        ~MemRateTest();

        /** Reads data from RAM and benchmarks the transfer rate.
        *  \param i64NumTransfers Specifies the number of times to read data from the entire buffer. 
        *  \return Returns the rate in GBps that the data was read from RAM.
        */
        float transfer(int64_t i64NumTransfers);

        /** Reads data from RAM for a specific period of time and benchmarks the transfer rate.
        *  \param i64NumSeconds_s Specifies the number of seconds to spend reading data from the entire buffer. 
        *  \return Returns the rate in GBps that the data was read from RAM.
        */
        float transferForLenghtOfTime(int64_t i64NumSeconds_s);
        
    protected:
        ///Number of threads to execute in parallel
        size_t m_ulNumThreads;

        ///Array of pointers to locations in memory to continuously read data from
        char ** m_ppMemArrays;

        ///Size of a single buffer
        size_t m_ulBufferSize_bytes; 
};

#endif