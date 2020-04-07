#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "UnitTest.hpp"

UnitTest::UnitTest() : m_iResult(0)
{
    cudaEventCreate(&m_eventHtoDStart);
    cudaEventCreate(&m_eventHtoDFinish);
    cudaEventCreate(&m_eventKernelStart);
    cudaEventCreate(&m_eventKernelFinish);
    cudaEventCreate(&m_eventDtoHStart);
    cudaEventCreate(&m_eventDtoHFinish);
}

UnitTest::~UnitTest()
{
    cudaEventDestroy(m_eventHtoDStart);
    cudaEventDestroy(m_eventHtoDFinish);
    cudaEventDestroy(m_eventKernelStart);
    cudaEventDestroy(m_eventKernelFinish);
    cudaEventDestroy(m_eventDtoHStart);
    cudaEventDestroy(m_eventDtoHFinish);
}


void UnitTest::run_test()
{
    /// The derived class provides a method to generate simulated input data for the operation.
    simulate_input();

    /// The input data is transferred to the device.
    cudaEventRecord(m_eventHtoDStart);
    transfer_HtoD();
    cudaEventRecord(m_eventHtoDFinish);
    cudaEventSynchronize(m_eventHtoDFinish); //call this to make sure the event is recorded properly!
    cudaEventElapsedTime(&m_fHtoDElapsedTime_ms, m_eventHtoDStart, m_eventHtoDFinish);


    /// The kernel under test is executed.
    cudaEventRecord(m_eventKernelStart);
    run_kernel();
    cudaEventRecord(m_eventKernelFinish);
    cudaEventSynchronize(m_eventKernelFinish);
    cudaEventElapsedTime(&m_fKernelElapsedTime_ms, m_eventKernelStart, m_eventKernelFinish);

    /// The processed data is tranferred to the host.
    cudaEventRecord(m_eventDtoHStart);
    transfer_DtoH();
    cudaEventRecord(m_eventDtoHFinish);
    cudaEventSynchronize(m_eventDtoHFinish);
    cudaEventElapsedTime(&m_fDtoHElapsedTime_ms, m_eventDtoHStart, m_eventDtoHFinish);

    /// The results are checked for correctness by the host.
    verify_output();
    ///\todo Accommodation should probably be made for tolerances - both as a pass/fail criterion,
    ///      and for reporting how much the GPU's results diverged from the CPU's.
}

int UnitTest::get_result()
{
    if (!m_iResult)
    {
        std::cerr << "UnitTest hasn't been run yet!" << std::endl;
        ///\todo Add a "name" parameter to the class, so that the warning that the test it hasn't run
        ///      yet actually has meaning to the user, e.g. if a suite of tests is being processed.
    }

    ///\retval  1 The test has passed.
    ///\retval -1 The test has failed.
    ///\retval  0 The test has not yet completed.
    return m_iResult;
}


float UnitTest::get_time()
{
    ///\todo Include some reporting around the sizes of memory transfers (i.e. data rates) and probably FLOPS too.
    std::cout << "HtoD:\t\t" << m_fHtoDElapsedTime_ms << " ms\n";
    std::cout << "Kernel:\t\t" << m_fKernelElapsedTime_ms << " ms\n";
    std::cout << "DtoH:\t\t" << m_fDtoHElapsedTime_ms << " ms\n";

    ///\return The time taken for the test to execute, including transfers to and from device, but excluding the
    ///        time taken by the CPU to generate simulated data or verify the output.
    return m_fHtoDElapsedTime_ms + m_fKernelElapsedTime_ms + m_fDtoHElapsedTime_ms;
}