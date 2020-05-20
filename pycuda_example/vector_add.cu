__global__ void kernel_vector_add(float *piVectorA, float *piVectorB, float *piVectorC, size_t ulVectorLength)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < ulVectorLength) //in case the size of the operation doesn't fit neatly into block size.
    {
        piVectorC[tid] = piVectorA[tid] + piVectorB[tid];
    }
}