#!/usr/bin/env python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# A big number. Just be careful not to overload the GPU.
vector_length = 200000000
block_size = 128
volume_knob = 256

# I'm using print() statements instead of comments here so that you can see what's happening when you run the script. It can take quite a while to finish...
print(f"Setting up memory on host and device for A, B (input) and C (output) vectors - {vector_length} elements in length...")
A_host = cuda.pagelocked_empty(int(vector_length), dtype=np.float32)
A_device = cuda.mem_alloc(A_host.nbytes)

B_host = cuda.pagelocked_empty(int(vector_length), dtype=np.float32)
B_device = cuda.mem_alloc(B_host.nbytes)

C_host = cuda.pagelocked_empty(int(vector_length), dtype=np.float32)
C_device = cuda.mem_alloc(C_host.nbytes)

print("Reading source file and JIT-compiling kernel...")
module = SourceModule(open("vector_add.cu").read())
vector_add_kernel = module.get_function("kernel_vector_add")

print(f"Populating host input vectors with random data with an amplitude of {volume_knob}...")
# We're using the [:] here so that the interpreter just updates the values within the numpy ndarray, rather
# than clobbering the specially-prepared pagelocked memory with a new ndarray.
A_host[:] = volume_knob*np.random.randn(vector_length)
B_host[:] = volume_knob*np.random.randn(vector_length)

print("Copying input data to device...")
cuda.memcpy_htod(A_device, A_host)
cuda.memcpy_htod(B_device, B_host)

# Calculate number of blocks
n_blocks = int(vector_length) // int(block_size) + 1

print("Executing kernel...")
vector_add_kernel(A_device, B_device, C_device, np.uint64(vector_length), # You can't pass a kernel a normal python int as an argument, it needs to be a numpy dtype.
            block=(block_size, 1, 1), # CUDA block and grid sizes can be 3-dimensional,
            grid=(n_blocks, 1, 1))    # but we're just using a 1D vector here.

print("Copying output data back to the host...")
cuda.memcpy_dtoh(C_host, C_device)

print(f"Host-side verification. It may take some time to check {vector_length} addtions...")
for i, (a, b, c) in enumerate(zip(A_host, B_host, C_host)):
    if c != a + b:
        print(f"Vector element {i} not correct! Got { a + b,}, expected {c}.")
        break
else:
    print(f"Vector-add of {vector_length} {C_host.dtype} elements completed successfully!")