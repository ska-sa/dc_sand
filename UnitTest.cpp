#include "UnitTest.hpp"

#include <iostream>
#include <cuda.h>

UnitTest::UnitTest() : m_iResult(0)
{
    cudaEventCreate(m_StartEvent);
    cudaEventCreate(m_StopEvent);
}

UnitTest::~UnitTest()
{
    cudaEventDestroy(m_StartEvent);
    cudaEventDestroy(m_StopEvent);
}


/**
 * \details Very few assumptions are made by the base class about what the derived class
 *          is going to test. The implementation of this function could be a single, simple
 *          kernel accompanied by memory transfers, or it could be as complex as multiple 
 *          interdependent streams.
 */
void UnitTest::run_test()
{
    /// The derived class provides a method to generate simulated input data for the operation.
    simulate_input();

    /// The operation as specified by the derived class is then executed, accompanied by some
    /// simple CUDA profiling infrastructure.
    cudaEventRecord(startEvent);
    run_operation();
    ///\todo It may be wise to make the concluding cudaDeviceSynchronize optional, so that it doesn't affect results on simple operations.
    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent);
    cudaEventElapsedTime(&m_fElapsedTime_ms, startEvent, stopEvent);

    /// The results are then checked for correctness by the host.
    /// This method is also implemented by the derived class.
    ///\todo Think about making the verification optional, selectable at run-time.
    ///\todo Should there be a parameter for the tolerance of the verification? Or should that be completely handled by the derived class?
    m_iResult = verify_output();
}

/** \details Returns the contents of m_iResult.
 *           This should have been updated by the derived class's implementation of verify_output().
 */
int UnitTest::get_result()
{
    /// The variable is initialised to zero by the constructor, if it's still zero it indicates that the test
    /// hasn't been run yet. This should give the user a warning.
    if (!m_iResult)
    {
        std::cerr << "UnitTest hasn't been run yet!" << std::endl;  ///\todo Add a "name" parameter so that the warning that it hasn't run yet actually has meaning to the user.
    }
    return m_iResult;
}