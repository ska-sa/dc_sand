#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define GPU_ERRCHK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true);

#endif