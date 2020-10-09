"""Digital Down-Converter for FEngine. This is a GPU implementation."""
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
from pycuda.compiler import SourceModule
import numpy as np
from numpy import genfromtxt
import cwg
from typing import List

import matplotlib.pyplot as plt

# from IPython import embed


fft_length = np.power(2, 12)
fir_size = 256
decimation_rate = 16

# A big number. Just be careful not to overload the GPU.
total_samples = np.power(2, 22)

input_samples = np.power(2, 22)  # 8192 * 10 #8192 * 4
print(f"Input samples is {input_samples}")
output_samples = input_samples / decimation_rate

ddc_coeff_filename = "../src/ddc_coeff_107MHz.csv"

volume_knob = 1


def _import_ddc_filter_coeffs(filename: str = "ddc_coeff_107MHz.csv"):
    """Import Digital Down Converter Filter Coefficients from file.

    Parameters
    ----------
    filename: str
        Digital Down Converter Filter Coefficients filename.
        The default name (if not passed filename): ddc_coeff_107MHz.csv
    Returns
    -------
    numpy ndarray of filter coefficients: type float.
    """
    print(f"Importing coefficients from {filename}")
    ddc_coeffs = genfromtxt(filename, delimiter=",")
    print(f"Imported {len(ddc_coeffs)} coefficients")
    return ddc_coeffs


# Calculate number of blocks
samples_per_block = 4096 #Limited by the amount of shared memory available per block
total_blocks = int(input_samples / samples_per_block)
threads_per_block = samples_per_block//decimation_rate
print(f"Total Blocks is {total_blocks}")


# I'm using print() statements instead of comments here so that you can see what's happening when you run the script. It can take quite a while to finish...
print(
    "Setting up memory on host and device for input_data, fir_coeffs (input) and data_downsampled_out (output) vectors"
)
data_in_host = cuda.pagelocked_empty(int(input_samples), dtype=np.float32)
data_in_device = cuda.mem_alloc(data_in_host.nbytes)

fir_coeffs_host = cuda.pagelocked_empty(int(fir_size), dtype=np.float32)
fir_coeffs_device = cuda.mem_alloc(fir_coeffs_host.nbytes)

data_downsampled_out_host = cuda.pagelocked_empty(int(output_samples * 2), dtype=np.float32)  # *2 for complex values
data_downsampled_out_device = cuda.mem_alloc(data_downsampled_out_host.nbytes)

data_debug_real_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
data_debug_real_out_device = cuda.mem_alloc(data_debug_real_out_host.nbytes)

data_debug_imag_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
data_debug_imag_out_device = cuda.mem_alloc(data_debug_imag_out_host.nbytes)

print("Reading source file and JIT-compiling kernel...")
module = SourceModule(open("ddc_kernel.cu").read())
ddc_kernel = module.get_function("kernel_ddc")

print(f"Populating host input vectors with random data with an amplitude of {volume_knob}...")
# We're using the [:] here so that the interpreter just updates the values within the numpy ndarray, rather
# than clobbering the specially-prepared pagelocked memory with a new ndarray.
fir_coeffs_host[:] = _import_ddc_filter_coeffs(filename=ddc_coeff_filename)

# Setup Mixing CW
osc_frequency = 100e6
print(f"Mixing CW is {osc_frequency/1e6}MHz")

# Setup input data

# freq = 107e6
# freq = 103343750
# Setup input data
cw_scale = 1
freq1 = 100e6
freq2 = 103343750
sampling_frequency = int(1712e6)
noise_scale = 0.00001

# Generate the CW for the test: CW for band centerde
cw1 = cwg.generate_carrier_wave(
    cw_scale=cw_scale,
    freq=freq1,
    sampling_frequency=sampling_frequency,
    num_samples=total_samples,
    noise_scale=noise_scale,
    complex=False,
)
# Generate the CW for the test: CW for arbitrary tone
cw2 = cwg.generate_carrier_wave(
    cw_scale=cw_scale,
    freq=freq2,
    sampling_frequency=sampling_frequency,
    num_samples=total_samples,
    noise_scale=noise_scale,
    complex=False,
)

# Combine both CW tones prior to putting through the DDC
# data = cw1 + cw2
data = cw1

# Compute number of iterations to work through data.
num_chunks = int(total_samples / input_samples)
print(f"Number of chunks is {num_chunks}")

# Test
# linear_input = np.linspace(0,total_samples-1,total_samples)
# print(f"linear is {linear_input[total_samples-1]}")
# cw = linear_input
gpu_decimated_data = []  # type: List[float]
# rebuild_debug = []
start = 0
end = 0

for chunk_number in range(num_chunks):
    # print(f"chunk_number is {chunk_number}")
    # print(f"Index range is {chunk_number*input_samples} to {chunk_number*input_samples + input_samples}")

    # Data input for the DDC
    # test = cw[chunk_number * input_samples : (chunk_number * input_samples + input_samples)]
    # print(f"test shape is {np.shape(test)}")
    # print(f"start value index is {(chunk_number * input_samples)} and data is {cw[chunk_number * input_samples]} and end value indx is {(chunk_number * input_samples + input_samples)} and data is {cw[chunk_number * input_samples + input_samples]}")

    # data_in_host[:] = cw[chunk_number * input_samples : (chunk_number * input_samples + input_samples-1)]
    data_in_host[0:input_samples] = data[chunk_number * input_samples : (chunk_number * input_samples + input_samples)]

    # print(f"Length of CW is {len(cw)}")

    print("Copying input data to device...")
    cuda.memcpy_htod(data_in_device, data_in_host)
    cuda.memcpy_htod(fir_coeffs_device, fir_coeffs_host)

    print("Executing kernel...")
    ddc_kernel(
        data_in_device,
        fir_coeffs_device,
        data_downsampled_out_device,
        np.float32(osc_frequency),
        np.int32(chunk_number),
        data_debug_real_out_device,
        data_debug_imag_out_device,  # You can't pass a kernel a normal python int as an argument, it needs to be a numpy dtype.
        block=(threads_per_block, 1, 1),  # CUDA block and grid sizes can be 3-dimensional,
        grid=(total_blocks, 1, 1),
    )  # but we're just using a 1D vector here.

    print("Copying output data back to the host...")
    cuda.memcpy_dtoh(data_downsampled_out_host, data_downsampled_out_device)
    # cuda.memcpy_dtoh(data_debug_real_out_host, data_debug_real_out_device)
    # cuda.memcpy_dtoh(data_debug_imag_out_host, data_debug_imag_out_device)

    # *** Debug ***
    skip = 256
    decimated_data = np.array(data_downsampled_out_host[0::2]) + np.array(data_downsampled_out_host[1::2]) * 1j

    # input_data_fft = np.abs(np.power(np.fft.rfft(data_in_host, axis=-1), 2))
    # decimated_cw_fft = np.abs(np.power(np.fft.fft(decimated_data[skip : (skip + fft_length)], axis=-1), 2))

    debug_cmplx = np.array(data_debug_real_out_host) + np.array(data_debug_imag_out_host) * 1j
    # debug_fft = np.abs(np.power(np.fft.fft(debug_cmplx[skip : (skip + fft_length)], axis=-1), 2))

    start = chunk_number * len(decimated_data)
    end = start + len(decimated_data)
    gpu_decimated_data[start:end] = decimated_data[:]

    start = chunk_number * len(debug_cmplx)
    end = start + len(debug_cmplx)
    # rebuild_debug[start:end] = debug_cmplx[:]

    if chunk_number == 1:

        # plt.figure(1)
        # plt.plot(np.real(decimated_data), ".-")

        # plt.figure(2)
        # plt.plot(np.imag(decimated_data), ".-")

        # plt.figure(1)
        # plt.plot(np.real(decimated_data))

        # plt.figure(2)
        # plt.plot(np.imag(decimated_data))

        plt.figure(3)
        plt.plot(data_debug_real_out_host, ".-")

        # plt.figure(4)
        # plt.plot(data_debug_imag_out_host, ".-")

        # # plt.figure(5)
        # # plt.semilogy(debug_fft)

        # # plt.figure(6)
        # # plt.semilogy(decimated_cw_fft)

        # plt.show()


# plt.figure(7)
# plt.plot(gpu_decimated_data)


gpu_decimated_data_fft = np.abs(np.power(np.fft.fft(gpu_decimated_data[skip : (skip + fft_length)], axis=-1), 2))

# # print(f"length of decimated data is {len(gpu_decimated_data)}")
plt.figure(5)
plt.semilogy(gpu_decimated_data_fft)

# plt.figure(8)
# plt.plot(data_debug_real_out_host, ".-")
# gpu_rebuild_debug_fft = np.abs(np.power(np.fft.fft(rebuild_debug[skip : (skip + fft_length)], axis=-1), 2))
# plt.figure(9)
# plt.semilogy(gpu_rebuild_debug_fft)

# # gpu_decimated_data_trunc = gpu_decimated_data[-fft_length:]

# # gpu_decimated_data_trunc_fft = np.abs(np.power(np.fft.fft(gpu_decimated_data_trunc, axis=-1), 2))

# # plt.figure(6)
# # plt.semilogy(gpu_decimated_data_trunc_fft)

plt.show()
