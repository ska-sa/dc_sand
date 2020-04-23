#ifndef __BERAMFORMER_KERNELS_CUH__
#define __BERAMFORMER_KERNELS_CUH__

#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void calculate_beamweights_naive(
                                struct timespec current_time, 
                                struct timespec ref_time,
                                struct delay_vals *dv, 
                                float* cplx_beamweights);

#endif