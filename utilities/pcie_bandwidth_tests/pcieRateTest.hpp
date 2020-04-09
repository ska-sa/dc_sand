#ifndef PCIE_RATE_TEST_H
#define PCIE_RATE_TEST_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

/**\class PcieRateTest
 * \brief   Measure the data rate between system RAM and a given PCIe device
 * \details The PcieRateTest class is a base class for testing the PCIe rate between host RAM and 
 *          a given PCIe device. This base class is abstract and requires a child class to implement
 *          driver specific transfers. A child class must support uni-directional transfers from
 *          Host to the Device and from the Device to Host. It must also implement a bidirectional 
 *          transfer. The transfer direction will be set on initialisation.
 * \example cudaPcieRateTest.hpp
 */
class PcieRateTest
{
    public:
        /// The default constructor is disabled.
        PcieRateTest() = delete;
        
        /**Constructs the PcieRateTestClass. Sets the PCIe device to use as well as the direction 
        *  transfers must occur. The transfers parameters are specified here. A single transfer 
        *  transfers a frame across the PCIe bus. A single buffer is allocated on the device - this 
        *  buffer consists of a number of frames.
        */
       PcieRateTest(int32_t i32DeviceId, int64_t i64NumFrames, int64_t i64FrameSizeBytes, bool bH2D, bool bD2H);
        
        /// The destructor must be virtual to ensure that derived classes' destructors are also called when the objects are destroyed.
        virtual ~PcieRateTest();

        /**Initiate a transfer across the PCIe bus. This function is PCIe device specific and must be implemented by the child class.
         * \param i64NumTransfers Specifies the number of frames to transfer across the bus
         */
        virtual float transfer(int64_t i64NumTransfers) = 0;

    protected:
        /// Device ID of the PCIe device to use. This is implementation specific. Different child classes will assign different device IDs to supported devices
        int32_t m_i32DeviceId;

        /// Number of frames to assigne to a single buffer
        int64_t m_i64NumFrames;

        /// Size of a single frame in bytes
        int64_t m_i64FrameSizeBytes;

        /// Specifies the size of the buffer in bytes to be allocated on the device
        int64_t m_i64DeviceBufferSize_bytes;

        /// Enable transfers from host to device
        bool m_bH2D;

        /// Enable transfers from device to host
        bool m_bD2H;
        
};

#endif