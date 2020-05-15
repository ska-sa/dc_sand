import pycuda.driver as cuda
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import numpy as np

tcm_size = 16

module = SourceModule(open("tc_dynamic_range.cu").read(),
                      options=["--ptxas-options=-v", f"-D TCM_SIZE={tcm_size}"],
                      arch="sm_75",
                      include_dirs=["/usr/local/cuda/include",],
                      no_extern_c=True )
simple_tc = module.get_function("simple_tc")
#do_something = module.get_function("do_something")
#do_nothing = module.get_function("do_nothing")

# A is just ones. 
a = np.ones((tcm_size, tcm_size), dtype=np.float16)
a_pin = cuda.register_host_memory(a)
a_gpu = cuda.mem_alloc(a_pin.nbytes)
cuda.memcpy_htod(a_gpu, a_pin)

# B will be mostly zeros.
b = np.zeros((tcm_size, tcm_size), dtype=np.float16)
# but we'll make one row with small numbers:
b[3,:] = 0.0625 #i.e. 1/16
for i in range(tcm_size):
    b[8,i] = 2*np.power(2,i)
b_pin = cuda.register_host_memory(b)
b_gpu = cuda.mem_alloc(a_pin.nbytes)
cuda.memcpy_htod(b_gpu, b_pin)

#print("B:")
#print(b[3,:])
#print(b[8,:])

c = np.empty((tcm_size, tcm_size), dtype=np.float32)
c_pin = cuda.register_host_memory(c)
c_gpu = cuda.mem_alloc(c_pin.nbytes)

simple_tc(a_gpu, b_gpu, c_gpu, block=(32,1,1), grid=(1,1,1))
#do_nothing( block=(1,1,1), grid=(1,1,1))
#do_something(a_gpu, block=(1,1,1), grid=(1,1,1))

cuda.memcpy_dtoh(c_pin, c_gpu)
#print("C:")
#print(c[0,:])

for i, (A,B) in enumerate(zip(b[3,:], b[8,:])):
    A = np.float32(A)
    B = np.float32(B)
    print(f"{i}: {A}\t{B}\t\t{c[0,i]}\t\t{c[0,i] - (A + B)}\t{np.log(B/A)/np.log(2)}")

