"""
tc_dynamic_range.py
A single, 16x16 matrix multiply performed on a tensor core, to test
the dynamic range of the arithmetic.
"""

import pycuda.driver as cuda
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import numpy as np

# TCM stands for "Tensor-core multiplier."
tcm_size = 16

def print_mat(matrix):
    """A quickly-and-dirty function to show me an entire 16x16 matrix."""
    # This little section prints 0 - 15 aligned with the columns of numbers to easily guide the eye.
    title_string = f"{'': >3}" # Python f-strings are a thing. Look them up they're probably the most sensible way to print things.
    for x in range(tcm_size):
        title_string += f"{x: >12}" # The >12 means right-align the number in a field 12 spaces wide.
    print(title_string)
    # The rest of the function just prints out the matrix
    for i in range(tcm_size):
        output_string = ""
        for j in range(tcm_size):
            output_string += f"{matrix[i,j]:10.5f}"
            output_string += "  "
        # with row-labels for easy reading.
        print(f"{i: >3}: {output_string}")

# This is the pycuda way of getting the kernel from a source file.
module = SourceModule(open("tc_dynamic_range.cu").read(),
                      options=["--ptxas-options=-v", f"-D TCM_SIZE={tcm_size}"], # PyCuda calls NVCC and you can pass compile options here.
                      arch="sm_75", # You do this if you need advanced features, e.g. tensor-cores. sm_75 gets us RTX 2080-level functionality.
                      include_dirs=["/usr/local/cuda/include",], # This because we use mma.h which isn't included in pycuda's default include dir.
                      no_extern_c=True ) #An explanation of this follows.
                      # PyCuda normally wraps the entirety of its included source in an extern "C" {...} block. This is to prevent the kernels from being compiled in a C++
                      # fashion, and allows the module.get_function() [see below] to actually find the function by its name (i.e. identifier).
                      # I need to use this option because if #include <mma.h> is inside the extern "C" block, then PyCuda kicks puppies and clubs baby seals.
                      # You need to manually put this block around your kernel for the functionality to work.
                      # NOTE: I am not sure whether this prevents us from using any advanced, C++-like features (such as classes, inheritance, polymorphism, etc) in kernels.
                      # Probably we won't need those things anyway.
simple_tc_matmul_kernel = module.get_function("simple_tc_matmul") # This looks for the "simple_tc_matmul" identifier in the output of nvcc.

# Set up the A matrix. Currently, it is mostly zeros.
a_host = cuda.pagelocked_zeros((tcm_size, tcm_size), dtype=np.float16)
# Manually insert some test data.
a_host[0,0] = 10
a_host[0,2] = 0.125
a_device = cuda.mem_alloc(a_host.nbytes)
cuda.memcpy_htod(a_device, a_host)

print("A:")
print_mat(a_host)

# B will be mostly zeros.
b_host = cuda.pagelocked_zeros((tcm_size, tcm_size), dtype=np.float16)
# Manually insert some test data.
b_host[0,0] = 1
b_host[2,0] = 0.125
b_device = cuda.mem_alloc(a_host.nbytes)
cuda.memcpy_htod(b_device, b_host)

print("B:")
print_mat(b_host)

c_host = cuda.pagelocked_empty((tcm_size, tcm_size), dtype=np.float32) # pagelocked_empty() doesn't bother to spend CPU time initialising the memory. There might be stuff in it if you aren't careful.
c_device = cuda.mem_alloc(c_host.nbytes)

# We launch the kernel with 32 threads because to do tensor-core multiplication we need an entire warp.
simple_tc_matmul_kernel(a_device, b_device, c_device, block=(32,1,1), grid=(1,1,1))

# Copy everything back once we're done.
cuda.memcpy_dtoh(c_host, c_device)

print("C:")
print_mat(c_host)
# The output can now be visually verified.
