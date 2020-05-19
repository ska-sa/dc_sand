#include <cuda_fp16.h>
#include <mma.h>

extern "C" {  //See the .py file for an explanation of why this is necessary.
    
__global__ void simple_tc_matmul(__half A[TCM_SIZE][TCM_SIZE], __half B[TCM_SIZE][TCM_SIZE], float C[TCM_SIZE][TCM_SIZE])
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, TCM_SIZE, TCM_SIZE, TCM_SIZE, half, row_major> a_frag;
    fragment<matrix_b, TCM_SIZE, TCM_SIZE, TCM_SIZE, half, row_major> b_frag;
    fragment<accumulator, TCM_SIZE, TCM_SIZE, TCM_SIZE, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    load_matrix_sync(a_frag, A[0], TCM_SIZE);
    load_matrix_sync(b_frag, B[0], TCM_SIZE);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(C[0], c_frag, TCM_SIZE, mem_row_major);
} 

}