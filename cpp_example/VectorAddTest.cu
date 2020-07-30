#include <iostream>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Utils.hpp"
#include "VectorAddTest.hpp"


/// The constructor adds a parameter which assigns the size of the vectors.
VectorAddTest::VectorAddTest(size_t uVectorLength) : m_uVectorLength(uVectorLength)
{
    /// Turns out that using pinned host memory is MUCH faster than the non-pinned variety.
    /// This sped things up by a factor of about 3 on my tests on qgpu02.
    GPU_ERRCHK(cudaMallocHost((void**)&m_piHVectorA, sizeof(*m_piHVectorA)*m_uVectorLength));
    GPU_ERRCHK(cudaMallocHost((void**)&m_piHVectorB, sizeof(*m_piHVectorB)*m_uVectorLength));
    GPU_ERRCHK(cudaMallocHost((void**)&m_piHVectorC, sizeof(*m_piHVectorC)*m_uVectorLength));

    GPU_ERRCHK(cudaMalloc((void **) &m_piDVectorA, m_uVectorLength*sizeof(*m_piDVectorA)));
    GPU_ERRCHK(cudaMalloc((void **) &m_piDVectorB, m_uVectorLength*sizeof(*m_piDVectorB)));
    GPU_ERRCHK(cudaMalloc((void **) &m_piDVectorC, m_uVectorLength*sizeof(*m_piDVectorC)));

    //Work out some kind of dimensionality for the kernel.
    //This example is nearly trivial since it needs virtually no memory, but this can often be
    //fairly critical for good utilisation of the GPU.
    m_ulBlockSize = 256;
    m_ulNumBlocks = (m_uVectorLength + m_ulBlockSize - 1) / m_ulBlockSize;  //round up.
}


/// The destructor just cleans up.
VectorAddTest::~VectorAddTest()
{
    GPU_ERRCHK(cudaFreeHost(m_piHVectorA));
    GPU_ERRCHK(cudaFreeHost(m_piHVectorB));
    GPU_ERRCHK(cudaFreeHost(m_piHVectorC));

    GPU_ERRCHK(cudaFree(m_piDVectorA));
    GPU_ERRCHK(cudaFree(m_piDVectorB));
    GPU_ERRCHK(cudaFree(m_piDVectorC));
}


/// Simulated input data in this case is a ramp up on one vector, a ramp down on the other. Each element of the sum should be the same.
void VectorAddTest::simulate_input()
{
    for (size_t i = 0; i < m_uVectorLength; i++)
    {
        m_piHVectorA[i] = i;
        m_piHVectorB[i] = m_uVectorLength - i;
    }
}


/// Simple transfer to device memory.
void VectorAddTest::transfer_HtoD()
{
    GPU_ERRCHK(cudaMemcpy(m_piDVectorA, m_piHVectorA, m_uVectorLength*sizeof(*m_piHVectorA), cudaMemcpyHostToDevice));
    GPU_ERRCHK(cudaMemcpy(m_piDVectorB, m_piHVectorB, m_uVectorLength*sizeof(*m_piHVectorA), cudaMemcpyHostToDevice));
}

#pragma region Original Kernel

//Kernel adds A and B, storing the result to C.
__global__ void kernel_vector_add(int *piVectorA, int *piVectorB, int *piVectorC, size_t ulVectorLength)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < ulVectorLength) //in case the size of the operation doesn't fit neatly into block size.
    {
        //for (int i = 0; i < 1000; i++) // To make the GPU spin for a while
        piVectorC[tid] = piVectorA[tid] + piVectorB[tid];
    }
}


void VectorAddTest::run_kernel()
{
    kernel_vector_add<<<m_ulNumBlocks, m_ulBlockSize>>>(m_piDVectorA, m_piDVectorB, m_piDVectorC, m_uVectorLength);
    GPU_ERRCHK(cudaGetLastError());
}

#pragma endregion 


void VectorAddTest::transfer_DtoH()
{
    GPU_ERRCHK(cudaMemcpy(m_piHVectorC, m_piDVectorC, m_uVectorLength*sizeof(*m_piHVectorC), cudaMemcpyDeviceToHost));
}


void VectorAddTest::verify_output()
{
    for (size_t i = 0; i < m_uVectorLength; i++)
    {
        if (m_piHVectorC[i] != (int) m_uVectorLength)
        {
            m_iResult = -1;
            std::cout << "Element " << i << " not equal. Expected " << m_uVectorLength << " but got " << m_piHVectorC[i] << "!\n";
            return;
        }
    }
    m_iResult = 1;
}
