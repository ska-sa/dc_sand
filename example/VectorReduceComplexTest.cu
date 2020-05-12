#include <iostream>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Utils.hpp"
#include "VectorReduceComplexTest.hpp"


/// The constructor adds a parameter which assigns the size of the vectors.
VectorReduceComplexTest::VectorReduceComplexTest(size_t uVectorLength) : m_uVectorLength(uVectorLength)
{
    GPU_ERRCHK(cudaMallocHost((void**)&m_piHVectorA, sizeof(*m_piHVectorA)*m_uVectorLength));
    GPU_ERRCHK(cudaMallocHost((void**)&m_piHVectorB, (sizeof(*m_piHVectorB)*(m_uVectorLength/2 + 1))));
    // GPU_ERRCHK(cudaMallocHost((void**)&m_piHVectorC, sizeof(*m_piHVectorC)*m_uVectorLength));

    GPU_ERRCHK(cudaMalloc((void **) &m_piDVectorA, m_uVectorLength*sizeof(*m_piDVectorA)));
    GPU_ERRCHK(cudaMalloc((void **) &m_piDVectorB, ((m_uVectorLength/2 + 1)*sizeof(*m_piDVectorB))));
    // GPU_ERRCHK(cudaMalloc((void **) &m_piDVectorC, m_uVectorLength*sizeof(*m_piDVectorC)));

    //Work out some kind of dimensionality for the kernel.
    //This example is nearly trivial since it needs virtually no memory, but this can often be
    //fairly critical for good utilisation of the GPU.
    m_ulBlockSize = 512;
    m_ulNumBlocks = ((m_uVectorLength + m_ulBlockSize - 1) / m_ulBlockSize) + 1;  //round up.
}


/// The destructor just cleans up.
VectorReduceComplexTest::~VectorReduceComplexTest()
{
    GPU_ERRCHK(cudaFreeHost(m_piHVectorA));
    GPU_ERRCHK(cudaFreeHost(m_piHVectorB));
    // GPU_ERRCHK(cudaFreeHost(m_piHVectorC));

    GPU_ERRCHK(cudaFree(m_piDVectorA));
    GPU_ERRCHK(cudaFree(m_piDVectorB));
    // GPU_ERRCHK(cudaFree(m_piDVectorC));
}


/// Simulated input data in this case is a ramp up on one vector, a ramp down on the other. Each element of the sum should be the same.
void VectorReduceComplexTest::simulate_input()
{
    for (size_t i = 0; i < m_uVectorLength; i++)
    {
        m_piHVectorA[i] = i;
    }
}


/// Simple transfer to device memory.
void VectorReduceComplexTest::transfer_HtoD()
{
    GPU_ERRCHK(cudaMemcpy(m_piDVectorA, m_piHVectorA, m_uVectorLength*sizeof(*m_piHVectorA), cudaMemcpyHostToDevice));
    // GPU_ERRCHK(cudaMemcpy(m_piDVectorB, m_piHVectorB, m_uVectorLength*sizeof(*m_piHVectorA), cudaMemcpyHostToDevice));
}


#pragma region Reduction Kernel

// Kernel computes square-then-sum of adjacent locations in memory
// - i.e. real | complex | real | complex | ...
// - But this needs to be done at a +=2 cadence!
__global__ void kernel_vector_reduce_complex(int *piVectorA, int *piVectorB, size_t ulVectorLength)
{
    int myId = threadIdx.x + blockIdx.x*blockDim.x;
    int realId = myId * 2;
    int imagId = realId + 1;
    int tid  = threadIdx.x;

    // Declare shared memory for magnitude to be assigned to
    // - Need to make sure that (ulVectorLength / 2) % 2 == 0
    extern __shared__ int sdata[];

    // Calculate magnitude^2 of complex data
    // - Don't need to square-root the result
    //   because it is squared later (for power calculation)
    sdata[tid] = (piVectorA[realId] * piVectorA[realId]) + (piVectorA[imagId] * piVectorA[imagId]);
    
    // Sync before moving onto summation component
    __syncthreads();


    for (size_t s = blockDim.x / 2; s > 0; s >>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Now, to handle the final reduction of tid == 0
    if (tid == 0)
    {
        // piVectorB[blockIdx.x] = sdata[myId];
        piVectorB[blockIdx.x] = sdata[0];
    }
}


void VectorReduceComplexTest::run_kernel()
{
    // const int maxThreadsPerBlock = 1024;

    kernel_vector_reduce_complex<<<m_ulNumBlocks, m_ulBlockSize, (m_uVectorLength/2 + 1)*sizeof(int)>>>(m_piDVectorA, m_piDVectorB, m_uVectorLength);
    GPU_ERRCHK(cudaGetLastError());
}

#pragma endregion


void VectorReduceComplexTest::transfer_DtoH()
{
    GPU_ERRCHK(cudaMemcpy(m_piHVectorB, m_piDVectorB, (m_uVectorLength/2 + 1)*sizeof(*m_piHVectorB), cudaMemcpyDeviceToHost));
}


void VectorReduceComplexTest::verify_output()
{
    
    for (size_t i = 0; i < m_uVectorLength; i++)
    {
        if (m_piHVectorB[i] > 0)
        {
            std::cout << "Output Vector[" << i << "] = " << m_piHVectorB[i] << ".\n";
        }
        
    }
}
