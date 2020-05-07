#!/usr/bin/env python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

vector_length = np.uint64(200000000)
block_size = 128


volume_knob = 256
A = volume_knob*np.random.randn(vector_length)
A = A.astype(np.float32)
A_pin = cuda.register_host_memory(A)  # There are actually several ways to do this. Must look into this.
A_gpu = cuda.mem_alloc(A_pin.nbytes)

B = volume_knob*np.random.randn(vector_length)
B = B.astype(np.float32)
B_pin = cuda.register_host_memory(B)
B_gpu = cuda.mem_alloc(B_pin.nbytes)

C = np.empty(vector_length, dtype=np.float32)
C_pin = cuda.register_host_memory(C)
C_gpu = cuda.mem_alloc(C_pin.nbytes)

module = SourceModule(open("vector_add.cu").read())
vector_add = module.get_function("kernel_vector_add")

cuda.memcpy_htod(A_gpu, A_pin)
cuda.memcpy_htod(B_gpu, B_pin)

n_blocks = int(vector_length) // int(block_size) + 1

vector_add(A_gpu, B_gpu, C_gpu, vector_length,
            block=(block_size, 1, 1),
            grid=(n_blocks, 1, 1))

cuda.memcpy_dtoh(C_pin, C_gpu)

for i, (a, b, c) in enumerate(zip(A, B, C)):
    if c != a + b:
        print("Vector element {} not correct! Got {}, expected {}.".format(i, a + b, c))
        break
else:
    print("Vector-add of {} {} elements completed successfully.".format(vector_length, C.dtype))