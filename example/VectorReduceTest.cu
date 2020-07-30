#include <iostream>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Utils.hpp"
#include "VectorReduceTest.hpp"


/// The constructor adds a parameter which assigns the size of the vectors.
VectorReduceTest::VectorReduceTest(size_t uVectorLength) : m_uVectorLength(uVectorLength)
{
    /// Turns out that using pinned host memory is MUCH faster than the non-pinned variety.
    /// This sped things up by a factor of about 3 on my tests on qgpu02.
    GPU_ERRCHK(cudaMallocHost((void**)&m_piHVectorA, sizeof(*m_piHVectorA)*m_uVectorLength));
    GPU_ERRCHK(cudaMallocHost((void**)&m_piHVectorB, sizeof(*m_piHVectorB)*m_uVectorLength));
    // GPU_ERRCHK(cudaMallocHost((void**)&m_piHVectorC, sizeof(*m_piHVectorC)*m_uVectorLength));

    GPU_ERRCHK(cudaMalloc((void **) &m_piDVectorA, m_uVectorLength*sizeof(*m_piDVectorA)));
    GPU_ERRCHK(cudaMalloc((void **) &m_piDVectorB, m_uVectorLength*sizeof(*m_piDVectorB)));
    // GPU_ERRCHK(cudaMalloc((void **) &m_piDVectorC, m_uVectorLength*sizeof(*m_piDVectorC)));

    //Work out some kind of dimensionality for the kernel.
    //This example is nearly trivial since it needs virtually no memory, but this can often be
    //fairly critical for good utilisation of the GPU.
    m_ulBlockSize = 256;
    m_ulNumBlocks = ((m_uVectorLength + m_ulBlockSize - 1) / m_ulBlockSize) + 1;  //round up.
}


/// The destructor just cleans up.
VectorReduceTest::~VectorReduceTest()
{
    GPU_ERRCHK(cudaFreeHost(m_piHVectorA));
    GPU_ERRCHK(cudaFreeHost(m_piHVectorB));
    // GPU_ERRCHK(cudaFreeHost(m_piHVectorC));

    GPU_ERRCHK(cudaFree(m_piDVectorA));
    GPU_ERRCHK(cudaFree(m_piDVectorB));
    // GPU_ERRCHK(cudaFree(m_piDVectorC));
}


/// Simulated input data in this case is a ramp up on one vector, a ramp down on the other. Each element of the sum should be the same.
void VectorReduceTest::simulate_input()
{
    for (size_t i = 0; i < m_uVectorLength; i++)
    {
        m_piHVectorA[i] = i;
    }
}


/// Simple transfer to device memory.
void VectorReduceTest::transfer_HtoD()
{
    GPU_ERRCHK(cudaMemcpy(m_piDVectorA, m_piHVectorA, m_uVectorLength*sizeof(*m_piHVectorA), cudaMemcpyHostToDevice));
    // GPU_ERRCHK(cudaMemcpy(m_piDVectorB, m_piHVectorB, m_uVectorLength*sizeof(*m_piHVectorA), cudaMemcpyHostToDevice));
}


#pragma region Reduction Kernel

//Kernel adds A and B, storing the result to C.
__global__ void kernel_vector_reduce(int *piVectorA, int *piVectorB, size_t ulVectorLength)
{
    int myId = threadIdx.x + blockIdx.x*blockDim.x;
    int tid  = threadIdx.x;

    for (size_t s = blockDim.x / 2; s > 0; s >>=1)
    {
        if (tid < s)
        {
            piVectorA[myId] += piVectorA[myId + s];
        }
        __syncthreads();
    }
    
    // Now, to handle the final reduction of tid == 0
    if (tid == 0)
    {
        piVectorB[blockIdx.x] = piVectorA[myId];
    }
}


void VectorReduceTest::run_kernel()
{
    // const int maxThreadsPerBlock = 1024;

    kernel_vector_reduce<<<m_ulNumBlocks, m_ulBlockSize>>>(m_piDVectorA, m_piDVectorB, m_uVectorLength);
    GPU_ERRCHK(cudaGetLastError());
}

#pragma endregion


void VectorReduceTest::transfer_DtoH()
{
    GPU_ERRCHK(cudaMemcpy(m_piHVectorB, m_piDVectorB, m_uVectorLength*sizeof(*m_piHVectorB), cudaMemcpyDeviceToHost));
}


void VectorReduceTest::verify_output()
{
    int sum = 0; // To hold running total of VectorA
    
    for (size_t i = 0; i < m_uVectorLength; i++)
    {
        sum = sum + m_piHVectorA[i];
	    // std::cout << "Running sum = " << sum << ".\n";
        if (m_piHVectorB[i] > 0)
        {
            std::cout << "Output Vector[" << i << "] = " << m_piHVectorB[i] << ".\n";
        }
        
    }
}
