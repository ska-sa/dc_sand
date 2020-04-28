#ifndef PCIE_RATE_TEST_H
#define PCIE_RATE_TEST_H

#include <cstdint> // For access to different integer types
#include <cstddef> //Access to size_t

/** \class      PcieRateTest
 *  \brief      Measure the data rate between system RAM and a given PCIe device
 *  \details    The PcieRateTest class is a base class for testing the PCIe rate between host RAM and 
 *              a given PCIe device. This base class is abstract and requires a child class to implement
 *              driver specific transfers. A child class must support uni-directional transfers from
 *              Host to the Device and from the Device to Host. It must also implement a bidirectional 
 *              transfer. The transfer direction will be set on initialisation.
 *  \example    cudaPcieRateTest.hpp
 */
class PcieRateTest
{
    public:
        /// The destructor must be virtual to ensure that derived classes' destructors are also called when the objects are destroyed.
        virtual ~PcieRateTest();

        /** Initiate a transfer across the PCIe bus. This function is PCIe device specific and must be implemented by the child class.
         *  \param i64NumTransfers Specifies the number of frames to transfer across the bus
         *  \return Returns the rate in Gbps that the data was transferred across the bus
         */
        virtual float transfer(int64_t i64NumTransfers) = 0;

        /** Initiate a transfer across the PCIe bus for a specific number of seconds. This function is PCIe device specific and must be implemented by the child class.
         *  \param i64NumTransfers Specifies the number of frames to transfer across the bus
         *  \return Returns the rate in Gbps that the data was transferred across the bus
         */
        virtual float transferForLengthOfTime(int64_t i64NumSeconds) = 0;

    protected:
        /** Constructs the PcieRateTest class. Sets the PCIe device to use as well as the direction 
        *   transfers must occur. The transfers parameters are specified here. A single transfer 
        *   transfers a frame across the PCIe bus. A single buffer is allocated on the device - this 
        *   buffer consists of a number of frames.
        */
        PcieRateTest(int32_t i32DeviceId, size_t ulNumFrames, int64_t ulFrameSizeBytes, bool bH2D, bool bD2H);

        /// Device ID of the PCIe device to use. This is implementation specific. Different child classes will assign different device IDs to supported devices
        int32_t m_i32DeviceId;

        /// Number of frames to assign to a single buffer
        size_t m_ulNumFrames;

        /// Size of a single frame in bytes
        int64_t m_ulFrameSizeBytes;

        /// Specifies the size of the buffer in bytes to be allocated on the device
        size_t m_ulDeviceBufferSize_bytes;

        /// Enable transfers from host to device
        bool m_bH2D;

        /// Enable transfers from device to host
        bool m_bD2H;
        
};

#endif