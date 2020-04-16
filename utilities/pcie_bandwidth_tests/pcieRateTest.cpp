#include "pcieRateTest.hpp"

PcieRateTest::PcieRateTest(int32_t i32DeviceId, size_t ulNumFrames, int64_t ulFrameSizeBytes,bool bH2D, bool bD2H):
    m_i32DeviceId(i32DeviceId),
    m_ulNumFrames(ulNumFrames),
    m_ulFrameSizeBytes(ulFrameSizeBytes),
    m_bH2D(bH2D),
    m_bD2H(bD2H),
    m_ulDeviceBufferSize_bytes(ulNumFrames*ulFrameSizeBytes)
{   
}

PcieRateTest::~PcieRateTest()
{
}
