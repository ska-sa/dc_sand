#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "VectorAddTest.hpp"

/// The constructor adds a parameter which assigns the size of the vectors.
VectorAddTest::VectorAddTest(size_t uVectorLength) : m_uVectorLength(uVectorLength)
{
    ///\todo Consider using cudaMallocHost.
    m_piHVectorA = new int[m_uVectorLength];
    m_piHVectorB = new int[m_uVectorLength];
    m_piHVectorC = new int[m_uVectorLength];

    cudaMalloc((void **) &m_piDVectorA, m_uVectorLength*sizeof(*m_piDVectorA));
    cudaMalloc((void **) &m_piDVectorB, m_uVectorLength*sizeof(*m_piDVectorB));
    cudaMalloc((void **) &m_piDVectorC, m_uVectorLength*sizeof(*m_piDVectorC));
}


/// The destructor just cleans up.
VectorAddTest::~VectorAddTest()
{
    delete[] m_piHVectorA;
    delete[] m_piHVectorB;
    delete[] m_piHVectorC;

    cudaFree(m_piDVectorA);
    cudaFree(m_piDVectorB);
    cudaFree(m_piDVectorC);
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
    cudaMemcpy(m_piDVectorA, m_piHVectorA, m_uVectorLength*sizeof(*m_piHVectorA), cudaMemcpyHostToDevice);
    cudaMemcpy(m_piDVectorB, m_piHVectorB, m_uVectorLength*sizeof(*m_piHVectorA), cudaMemcpyHostToDevice);
}


//Kernel adds A and B, storing the result to C.
__global__ void kernel_vector_add(int *A, int *B, int *C, size_t N)
{
    int tid = blockIdx.x*blockDimx.x + threadIdx.x;
    if (tid < N) //in case the size of the operation doesn't fit neatly into block size.
    {
        C[tid] = A[tid] + B[tid];
    }
}


void VectorAddTest::run_kernel()
{
    //Work out some kind of dimensionality for the kernel.
    //This example is nearly trivial since it needs virtually no memory, but this can often be
    //fairly critical for good utilisation of the GPU.
    int blockSize = 256;
    int numBlocks = (m_uVectorLength + blockSize - 1) / blockSize;
    kernel_vector_add<< numBlocks, blockSize >>>(m_piDVectorA, m_piDVectorB, m_piDVectorC, m_uVectorLength);
}


void VectorAddTest::transfer_DtoH()
{
    cudaMemcpy(m_piHVectorC, m_piDVectorC, m_uVectorLength*sizeof(*m_piHVectorC), cudaMemcpyDeviceToHost);
}


void VectorAddTest::verify_output()
{
    for (size_t i = 0; i < m_uVectorLength; i++)
    {
        if (m_piHVectorC[i] != (int) m_uVectorLength)
        {
            m_iResult = -1;
            return;
        }
    }
    m_iResult = 1;
}