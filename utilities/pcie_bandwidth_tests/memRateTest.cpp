#include "memRateTest.hpp"
#include "memRateTest_asm.h"
#include <chrono>
#include <omp.h>
#include <sys/mman.h>

MemRateTest::MemRateTest(size_t ulNumThreads, size_t ulBufferSize_bytes,bool useHugePages): 
    m_ulNumThreads(ulNumThreads),
    m_ulBufferSize_bytes(ulBufferSize_bytes)
{
    //Create a buffer for every thread
    m_ppMemArrays = new char*[ulNumThreads];
    //Performs this allocation in parallel - this is not necessary for performance testing as it is not included in the timing
    #pragma omp parallel for
    for (size_t i = 0; i < ulNumThreads; i++)
    {
        if(useHugePages)
        {
            m_ppMemArrays[i] = allocate(m_ulBufferSize_bytes,1);
        }else{
            m_ppMemArrays[i] = allocate(m_ulBufferSize_bytes,0);
        }
    }
}

MemRateTest::~MemRateTest(){
    //Free all buffers
    //Performs this in parallel - this is not necessary for performance testing as it is not included in the timing
    #pragma omp parallel for
    for (size_t i = 0; i < m_ulNumThreads; i++)
    {
        munmap(m_ppMemArrays[i],m_ulBufferSize_bytes);
    }
    delete m_ppMemArrays;
}

float MemRateTest::transfer(int64_t i64NumTransfers){    
    std::chrono::duration<double> timeElapsed_s[m_ulNumThreads];
    //Launches a number of threads performing memory reads simultaneously
    #pragma omp parallel for
    for (size_t i = 0; i < m_ulNumThreads; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        ScanWrite256PtrUnrollLoop(m_ppMemArrays[i],m_ulBufferSize_bytes,i64NumTransfers);
        auto now = std::chrono::high_resolution_clock::now();
        timeElapsed_s[i] = now - start;
    }

    //Calculates the combined data rate of all the threads
    float fCombinedRate_GBps = 0;
    for (size_t i = 0; i < m_ulNumThreads; i++)
    {
        float fRate = i64NumTransfers*m_ulBufferSize_bytes/1000.0/1000.0/1000.0/timeElapsed_s[i].count();
        fCombinedRate_GBps+=fRate;
    }
    return fCombinedRate_GBps;
}

float MemRateTest::transferForLenghtOfTime(int64_t i64NumSeconds_s){
    std::chrono::duration<double> timeElapsed_s[m_ulNumThreads];
    int64_t i64TransferSize_bytes[m_ulNumThreads];
    int i64NumTransfers = 25;
    //Launches a number of threads performing memory reads simultaneously
    #pragma omp parallel for 
    for (size_t i = 0; i < m_ulNumThreads; i++)
    {
        int32_t iterations=0;
        auto start = std::chrono::high_resolution_clock::now();
        //Read data from RAM. Read keeps looping until i64NumSeconds_s have passed
        do
        {
            ScanWrite256PtrUnrollLoop(m_ppMemArrays[i],m_ulBufferSize_bytes,i64NumTransfers);
            auto now = std::chrono::high_resolution_clock::now();
            iterations++;
            timeElapsed_s[i] = now - start;
        } while (timeElapsed_s[i].count() < i64NumSeconds_s);
        i64TransferSize_bytes[i] = (int64_t)iterations*(int64_t)i64NumTransfers*(int64_t)m_ulBufferSize_bytes;
    }

    //Calculates the combined data rate of all the threads
    float fCombinedRate_GBps = 0;
    for (size_t i = 0; i < m_ulNumThreads; i++)
    {
        float fRate = i64TransferSize_bytes[i]/1000.0/1000.0/1000.0/timeElapsed_s[i].count();
        fCombinedRate_GBps+=fRate;
    }
    return fCombinedRate_GBps;
}