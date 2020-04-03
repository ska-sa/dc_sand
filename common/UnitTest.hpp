#ifndef __TEST_HPP__
#define __TEST_HPP__


/**\class UnitTest
 * \brief   Automated unit tests
 * \details The assumption of the UnitTest class is that a single kernel will be tested in isolation,
 *          with the intent to understand the performance of the kernel. Each kernel should
 *          be tested on its own, as most of our kernels should be parallel enough to use 100% of the
 *          GPU's compute capacity.  The data transfers are also profiled, the usefulness of this is
 *          determined by the author of the derived class.
 * \example VectorAddTest.hpp
 */
class UnitTest
{
    public:
        /// The destructor must be virtual to ensure that derived classes' destructors are also called when the objects are destroyed.
        virtual ~UnitTest();

        /// Execute the test.
        void  run_test();
        /// Retrieve the test result.
        int   get_result();
        /// Retrieve the amount of time that the test took.
        float get_time();
    
    protected:
        /// The constructor is protected as a reminder that we can't instantiate a pure virtual class directly.
        UnitTest();

        /// Generate simulated input data.
        virtual void simulate_input() = 0;

        /// Transfer the simulated data to device memory.
        virtual void transfer_HtoD()  = 0;

        /// Execute the kernel.
        virtual void run_kernel()  = 0;

        /// Transfer the processed data from device memory back to the host.
        virtual void transfer_DtoH() = 0;

        /// Verify the correctness of the output data.
        virtual void verify_output()  = 0;

        /// Stores the result of the test.
        int   m_iResult;

    private:
        /// Timing for the start of the HtoD memory transfer.
        cudaEvent_t m_eventHtoDStart;
        /// Timing for the finish of the HtoD memory transfer.
        cudaEvent_t m_eventHtoDFinish;
        /// Time duration (ms) of the HtoD memory transfer.
        float m_fHtoDElapsedTime_ms;

        /// Timing for the start of the kernel execution.
        cudaEvent_t m_eventKernelStart;
        /// Timing for the finish of the kernel execution.
        cudaEvent_t m_eventKernelFinish;
        /// Time duration (ms) of the kernel execution.
        float m_fKernelElapsedTime_ms;

        /// Timing for the start of the DtoH memory transfer.
        cudaEvent_t m_eventDtoHStart;
        /// Timing for the finish of the DtoH memory transfer.
        cudaEvent_t m_eventDtoHFinish;
        /// Time duration (ms) of the DtoH memory transfer.
        float m_fDtoHElapsedTime_ms;
}

#endif