#include <cstdint>
#include <iostream>
#include <cuda.h>

#include "Utils.hpp"

///A quick check to report errors arising from CUDA functions.
void gpu_assert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << ":" << line << "\n";
      if (abort)
        exit(code);
   }
}
