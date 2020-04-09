#ifndef PCIE_RATE_TEST_H
#define PCIE_RATE_TEST_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

class PcieRateTest
{
    public:
        PcieRateTest() = delete;
        
        PcieRateTest(int32_t i32DeviceId, int64_t i64NumFrames, int64_t i64FrameSizeBytes, int64_t i64NumTransfers ,bool bH2D, bool bD2H);
        
        virtual ~PcieRateTest();

        virtual float transfer() = 0;

    protected:
        int32_t m_i32DeviceId;

        int64_t m_i64NumFrames;
        int64_t m_i64FrameSizeBytes;
        int64_t m_i64NumTransfers;
        int64_t m_i64ArraySize_bytes;

        bool m_bH2D;
        bool m_bD2H;

};

#endif