#include "Test.hpp"

#include <cuda.h>

//TODO: Doxygen documentation for the function.
void Test::run_test()
{
    // Generate simulated data for the 
    simulate_input();

    // Profiling.
    // We could in principle make these member variables but then we'd need a constructor.
    cudaEvent_t startEvent;
    cudaEventCreate(&startEvent);
    cudaEvent_t stopEvent;
    cudaEventCreate(&stopEvent);

    //Execute the test.
    cudaEventRecord(startEvent);
    run_operation();
    cudaDeviceSynchronzse();
    cudaEventRecord(stopEvent);

    // Determine the time elapsed
    cudaEventElapsedTime(&m_fElapsedTime_ms, startEvent, stopEvent);

    // Check results for correctness.
    // TODO:
    // - think about making it optional. If you've checked for correctness you mightn't want to run this every time.
    // - think about adding a parameter for tolerance, because the reduced-precision arithmetic on the GPU won't
    //   necessarily be bit-perfect compared with what the CPU calculates.
    m_iResult = verify_output();

    // Cleanup.
    // Could possibly move to destructor if these become member variables.
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}