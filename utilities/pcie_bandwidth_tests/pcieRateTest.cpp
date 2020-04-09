#include "pcieRateTest.h"

PcieRateTest::PcieRateTest(int32_t i32DeviceId, int64_t i64NumFrames, int64_t i64FrameSizeBytes, int64_t i64NumTransfers ,bool bH2D, bool bD2H):
    m_i32DeviceId(i32DeviceId),
    m_i64NumFrames(i64NumFrames),
    m_i64FrameSizeBytes(i64FrameSizeBytes),
    m_i64NumTransfers(i64NumTransfers),
    m_bH2D(bH2D),
    m_bD2H(bD2H)
{
    
}

PcieRateTest::~PcieRateTest(){

}
