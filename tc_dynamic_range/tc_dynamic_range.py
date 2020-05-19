import pycuda.driver as cuda
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import numpy as np

# TCM stands for "Tensor-core multiplier."
tcm_size = 16

def print_mat(matrix):
    """A quickly-and-dirty function to show me an entire 16x16 matrix."""
    title_string = f"{'': >3}"
    for x in range(tcm_size):
        title_string += f"{x: >12}"
    print(title_string)
    for i in range(tcm_size):
        output_string = ""
        for j in range(tcm_size):
            output_string += f"{matrix[i,j]:10.5f}"
            output_string += "  "
        print(f"{i: >3}: {output_string}")

# This is the pycuda way of getting the kernel from a source file.
# Technically it wants a raw string, but I like to keep files separate. So I use this method to read
# a file into a string directly into the source module's __init__ function.
module = SourceModule(open("tc_dynamic_range.cu").read(),
                      options=["--ptxas-options=-v", f"-D TCM_SIZE={tcm_size}"],
                      arch="sm_75",
                      include_dirs=["/usr/local/cuda/include",],
                      no_extern_c=True )
#get the kernel from the module
simple_tc_matmul_kernel = module.get_function("simple_tc_matmul")

# Set up the A matrix. Currently, it is mostly zeros.
a_host = cuda.pagelocked_zeros((tcm_size, tcm_size), dtype=np.float16)
a_host[0,0] = 10
a_host[0,2] = 0.125
a_device = cuda.mem_alloc(a_host.nbytes)
cuda.memcpy_htod(a_device, a_host)

print("A:")
print_mat(a_host)

# B will be mostly zeros.
b_host = cuda.pagelocked_zeros((tcm_size, tcm_size), dtype=np.float16)
b_host[0,0] = 1
b_host[2,0] = 0.125
b_device = cuda.mem_alloc(a_host.nbytes)
cuda.memcpy_htod(b_device, b_host)

print("B:")
print_mat(b_host)

c_host = cuda.pagelocked_zeros((tcm_size, tcm_size), dtype=np.float32)
c_device = cuda.mem_alloc(c_host.nbytes)

# We launch the kernel with 32 threads because to do tensor-core multiplication we need an entire warp.
simple_tc_matmul_kernel(a_device, b_device, c_device, block=(32,1,1), grid=(1,1,1))

# Copy everything back once we're done.
cuda.memcpy_dtoh(c_host, c_device)

print("C:")
print_mat(c_host)

