import pycuda.driver as cuda
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import numpy as np

# TCM stands for "Tensor-core multiplier."
tcm_size = 16

def print_mat(matrix):
    """A quickly-and-dirty function to show me an entire 16x16 matrix."""
    title_string = f"{'': >3}"
    for x in range(3):
        title_string += f"{x: >12}"
    print(title_string)
    for i in range(3):
        output_string = ""
        for j in range(3):
            output_string += f"{matrix[i,j]:10.5f}"
            output_string += "  "
        print(f"{i: >3}: {output_string}")

# This is the py-cuda way of getting the kernel from a source file.
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
a = np.zeros((tcm_size, tcm_size), dtype=np.float16)
a[0,0] = 512
a[0,2] = 0.125
# Columns 3 and 8 will be ones to match up with equivalent rows in B. 
#a[:,3] = 1.0
#a[:,8] = 1.0
a_pin = cuda.register_host_memory(a)
a_gpu = cuda.mem_alloc(a_pin.nbytes)
cuda.memcpy_htod(a_gpu, a_pin)

print("A:")
print_mat(a)

# B will be mostly zeros.
b = np.zeros((tcm_size, tcm_size), dtype=np.float16)
b[0,0] = 512
b[2,0] = 0.125
# Row 3 will be a small number
#b[3,:] = 0.0625 #i.e. 1/16
#Row 8 starts at 2 and doubles until it reaches 65535 (or tries to)
#for i in range(tcm_size):
#    b[8,i] = 2*np.power(2,i)
b_pin = cuda.register_host_memory(b)
b_gpu = cuda.mem_alloc(a_pin.nbytes)
cuda.memcpy_htod(b_gpu, b_pin)

print("B:")
print_mat(b)

c = np.empty((tcm_size, tcm_size), dtype=np.float32)
c_pin = cuda.register_host_memory(c)
c_gpu = cuda.mem_alloc(c_pin.nbytes)

# We launch the kernel with 32 threads because to do tensor-core multiplication we need an entire warp.
simple_tc_matmul_kernel(a_gpu, b_gpu, c_gpu, block=(32,1,1), grid=(1,1,1))

cuda.memcpy_dtoh(c_pin, c_gpu)
context.synchronize()

print("C:")
print_mat(c)

