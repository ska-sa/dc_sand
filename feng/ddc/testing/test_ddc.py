"""Test Process for Digital Down Conversion."""
import ddc
import cwg
import numpy as np
import pytest
import logging

import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
from pycuda.compiler import SourceModule

# from numpy import genfromtxt
from typing import List


@pytest.fixture
def DDC_fixture():
    """Create DDC test object for pytest."""
    decimation_factor = 16
    sampling_frequency = 1712e6
    ddc_coeff_filename = "../src/ddc_coeff_107MHz.csv"
    return ddc.DigitalDownConverter(
        decimation_factor=decimation_factor,
        sampling_frequency=sampling_frequency,
        ddc_coeff_filename=ddc_coeff_filename,
    )


def test_run_ddc_center_cw(DDC_fixture):
    """Test to verify correct translation of center frequency CW down to baseband (DC).

    The purpose of this test is to check the correct translation of the center frequency CW.

    Test Overview:
    This test will inject a single CW tone in the band center.
    The intention is to check for the correct position of the CW tone in frequency domain.
    The test CW tone is 100MHz for a band center of 100MHz.
    The test will look for the following:
    a) Is the test tone correctly placed in it's respective frequency channel?

    """
    # Specify Channel threshold to decide if energy present is significant
    channel_threshold = 1e5

    # Generate CW to test DDC
    cw_scale = 1
    freq = 100e6
    sampling_frequency = 1712e6
    noise_scale = 0
    mixing_freq = 100e6
    fft_length = np.power(2, 15)
    total_samples = fft_length * DDC_fixture.decimation_factor * 2

    # Generate the CW for the test
    data = cwg.generate_carrier_wave(
        cw_scale=cw_scale,
        freq=freq,
        sampling_frequency=sampling_frequency,
        num_samples=total_samples,
        noise_scale=noise_scale,
        complex=False,
    )

    """ CPU DDC Compute """
    # Run the DDC on test CW
    cpu_decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft. In this test the mixing CW is the same frequency as the test CW.
    cpu_decimated_data_trunc = cpu_decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    cpu_ddc_fft = np.power(np.fft.fft(cpu_decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    cpu_above_threshold_channels = np.where(np.abs(cpu_ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 1.
    assert len(cpu_above_threshold_channels) == 1

    # Specify expected channel where we expect the translation to occur. In this test it should be the DC bin(0)
    expected_translation_center_channel = np.floor(
        (freq - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    ddc_translation_center_channel = cpu_above_threshold_channels[0][0]
    assert ddc_translation_center_channel == expected_translation_center_channel

    """ GPU DDC Compute """
    fir_size = 256

    # A big number. Just be careful not to overload the GPU.
    input_samples = total_samples
    output_samples = input_samples / DDC_fixture.decimation_factor

    # Calculate number of blocks
    samples_per_block = fir_size * 16
    total_blocks = int(input_samples / samples_per_block)
    print(f"Total Blocks is {total_blocks}")

    # I'm using print() statements instead of comments here so that you can see what's happening when you run the script. It can take quite a while to finish...
    print(
        "Setting up memory on host and device for input_data, fir_coeffs (input) and data_downsampled_out (output) vectors"
    )
    data_in_host = cuda.pagelocked_empty(int(input_samples), dtype=np.float32)
    data_in_device = cuda.mem_alloc(data_in_host.nbytes)

    fir_coeffs_host = cuda.pagelocked_empty(int(fir_size), dtype=np.float32)
    fir_coeffs_device = cuda.mem_alloc(fir_coeffs_host.nbytes)

    data_downsampled_out_host = cuda.pagelocked_empty(
        int(output_samples * 2), dtype=np.float32
    )  # *2 for complex values
    data_downsampled_out_device = cuda.mem_alloc(data_downsampled_out_host.nbytes)

    data_debug_real_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
    data_debug_real_out_device = cuda.mem_alloc(data_debug_real_out_host.nbytes)

    data_debug_imag_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
    data_debug_imag_out_device = cuda.mem_alloc(data_debug_imag_out_host.nbytes)

    print("Reading source file and JIT-compiling kernel...")
    module = SourceModule(open("ddc_kernel.cu").read())
    ddc_kernel = module.get_function("kernel_ddc")

    # We're using the [:] here so that the interpreter just updates the values within the numpy ndarray, rather
    # than clobbering the specially-prepared pagelocked memory with a new ndarray.
    fir_coeffs_host[:] = DDC_fixture.ddc_filter_coeffs

    # Compute number of iterations to work through data.
    num_chunks = int(total_samples / input_samples)
    print(f"Num chunks is {num_chunks}")

    gpu_decimated_data = []  # type: List[float]
    start = 0
    end = 0

    for chunk_number in range(num_chunks):
        data_in_host[0:input_samples] = data[
            chunk_number * input_samples : (chunk_number * input_samples + input_samples)
        ]

        print("Copying input data to device...")
        cuda.memcpy_htod(data_in_device, data_in_host)
        cuda.memcpy_htod(fir_coeffs_device, fir_coeffs_host)

        print("Executing kernel...")
        ddc_kernel(
            data_in_device,
            fir_coeffs_device,
            data_downsampled_out_device,
            np.float32(mixing_freq),
            np.int32(chunk_number),
            data_debug_real_out_device,
            data_debug_imag_out_device,  # You can't pass a kernel a normal python int as an argument, it needs to be a numpy dtype.
            block=(fir_size, 1, 1),  # CUDA block and grid sizes can be 3-dimensional,
            grid=(total_blocks, 1, 1),
        )  # but we're just using a 1D vector here.

        print("Copying output data back to the host...")
        cuda.memcpy_dtoh(data_downsampled_out_host, data_downsampled_out_device)
        # cuda.memcpy_dtoh(data_debug_real_out_host, data_debug_real_out_device)
        # cuda.memcpy_dtoh(data_debug_imag_out_host, data_debug_imag_out_device)

        decimated_data = np.array(data_downsampled_out_host[0::2]) + np.array(data_downsampled_out_host[1::2]) * 1j

        start = chunk_number * len(decimated_data)
        end = start + len(decimated_data)
        gpu_decimated_data[start:end] = decimated_data[:]

    # Extract length of data for fft. In this test the mixing CW is the same frequency as the test CW.
    gpu_decimated_data_trunc = gpu_decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    gpu_ddc_fft = np.power(np.fft.fft(gpu_decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    gpu_above_threshold_channels = np.where(np.abs(gpu_ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 1.
    assert len(gpu_above_threshold_channels) == 1

    # Specify expected channel where we expect the translation to occur. In this test it should be the DC bin(0)
    expected_translation_center_channel = np.floor(
        (freq - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    gpu_ddc_translation_center_channel = gpu_above_threshold_channels[0][0]
    assert gpu_ddc_translation_center_channel == expected_translation_center_channel


def test_run_ddc_dual_cw(DDC_fixture):
    """Test to verify correct translation of center frequecny CW and additional in-band CW.

    The purpose of this test is to check the correct translation of the center frequency CW as well as
    a second arbitrary CW tone placed mid-band.

    Test Overview:
    This test will inject two CW tones - one in the band center and one offset from band center.
    The intention is to check for the correct position of the two CW tones in frequency domain.
    The two test CW tones are 100MHz and 103.343750MHz for a band center of 100MHz.
    The offset test CW tone is chosen to fall in the center of the frequency channel.
    The test will look for the following:
    a) Are the two test tones correctly placed in their respective frequency channels?

    """
    # Specify Channel threshold to decide if energy present is significant
    channel_threshold = 1e5

    # Generate CW to test DDC
    cw_scale = 1
    freq1 = 100e6
    freq2 = 103343750
    sampling_frequency = 1712e6
    fft_length = np.power(2, 10)
    total_samples = fft_length * DDC_fixture.decimation_factor * 70
    noise_scale = 0.00001
    mixing_freq = freq1

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
    data = cw1 + cw2

    """ CPU DDC Compute """
    # Run the DDC on test CW
    cpu_decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft
    cpu_decimated_data_trunc = cpu_decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    cpu_ddc_fft = np.power(np.fft.fft(cpu_decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    cpu_above_threshold_channels = np.where(np.abs(cpu_ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 2.
    assert np.size(cpu_above_threshold_channels) == 2

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = np.floor(
        (freq1 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )
    expected_translation_off_center_channel = np.floor(
        (freq2 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    assert cpu_above_threshold_channels[0][0] == expected_translation_center_channel
    assert cpu_above_threshold_channels[0][1] == expected_translation_off_center_channel

    """ GPU DDC Compute """
    fir_size = 256

    # A big number. Just be careful not to overload the GPU.
    input_samples = total_samples
    output_samples = input_samples / DDC_fixture.decimation_factor

    # Calculate number of blocks
    samples_per_block = fir_size * 16
    total_blocks = int(input_samples / samples_per_block)
    print(f"Total Blocks is {total_blocks}")

    # I'm using print() statements instead of comments here so that you can see what's happening when you run the script. It can take quite a while to finish...
    print(
        "Setting up memory on host and device for input_data, fir_coeffs (input) and data_downsampled_out (output) vectors"
    )
    data_in_host = cuda.pagelocked_empty(int(input_samples), dtype=np.float32)
    data_in_device = cuda.mem_alloc(data_in_host.nbytes)

    fir_coeffs_host = cuda.pagelocked_empty(int(fir_size), dtype=np.float32)
    fir_coeffs_device = cuda.mem_alloc(fir_coeffs_host.nbytes)

    data_downsampled_out_host = cuda.pagelocked_empty(
        int(output_samples * 2), dtype=np.float32
    )  # *2 for complex values
    data_downsampled_out_device = cuda.mem_alloc(data_downsampled_out_host.nbytes)

    data_debug_real_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
    data_debug_real_out_device = cuda.mem_alloc(data_debug_real_out_host.nbytes)

    data_debug_imag_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
    data_debug_imag_out_device = cuda.mem_alloc(data_debug_imag_out_host.nbytes)

    print("Reading source file and JIT-compiling kernel...")
    module = SourceModule(open("ddc_kernel.cu").read())
    ddc_kernel = module.get_function("kernel_ddc")

    # We're using the [:] here so that the interpreter just updates the values within the numpy ndarray, rather
    # than clobbering the specially-prepared pagelocked memory with a new ndarray.
    fir_coeffs_host[:] = DDC_fixture.ddc_filter_coeffs

    # Compute number of iterations to work through data.
    num_chunks = int(total_samples / input_samples)
    print(f"Num chunks is {num_chunks}")

    gpu_decimated_data = []  # type: List[float]
    start = 0
    end = 0

    for chunk_number in range(num_chunks):
        data_in_host[0:input_samples] = data[
            chunk_number * input_samples : (chunk_number * input_samples + input_samples)
        ]

        print("Copying input data to device...")
        cuda.memcpy_htod(data_in_device, data_in_host)
        cuda.memcpy_htod(fir_coeffs_device, fir_coeffs_host)

        print("Executing kernel...")
        ddc_kernel(
            data_in_device,
            fir_coeffs_device,
            data_downsampled_out_device,
            np.float32(mixing_freq),
            np.int32(chunk_number),
            data_debug_real_out_device,
            data_debug_imag_out_device,  # You can't pass a kernel a normal python int as an argument, it needs to be a numpy dtype.
            block=(fir_size, 1, 1),  # CUDA block and grid sizes can be 3-dimensional,
            grid=(total_blocks, 1, 1),
        )  # but we're just using a 1D vector here.

        print("Copying output data back to the host...")
        cuda.memcpy_dtoh(data_downsampled_out_host, data_downsampled_out_device)
        # cuda.memcpy_dtoh(data_debug_real_out_host, data_debug_real_out_device)
        # cuda.memcpy_dtoh(data_debug_imag_out_host, data_debug_imag_out_device)

        decimated_data = np.array(data_downsampled_out_host[0::2]) + np.array(data_downsampled_out_host[1::2]) * 1j

        start = chunk_number * len(decimated_data)
        end = start + len(decimated_data)
        gpu_decimated_data[start:end] = decimated_data[:]

    # Extract length of data for fft
    gpu_decimated_data_trunc = gpu_decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    gpu_ddc_fft = np.power(np.fft.fft(gpu_decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    gpu_above_threshold_channels = np.where(np.abs(gpu_ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 2.
    assert np.size(gpu_above_threshold_channels) == 2

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = np.round(
        (freq1 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )
    expected_translation_off_center_channel = np.round(
        (freq2 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    assert gpu_above_threshold_channels[0][0] == expected_translation_center_channel
    assert gpu_above_threshold_channels[0][1] == expected_translation_off_center_channel


def test_run_ddc_bandedge_cw(DDC_fixture):
    """Test to verify correct translation of two in-band CW tones at band edges.

    The purpose of this test is to check the correct translation of the two CW tones placed at the band edges.
    This will differ depending on the NarrowBand mode to be tested.

    Test Overview:
    This test will inject two CW tones - one on each of the band edges.
    The intention is to check for the correct position of the two CW tones in frequency domain.
    The two test CW tones are 51.019287109375MHz and 148.980712890625MHz for a band center of 100MHz.
    These two test CW tones are chosen to fall in the center of their respective bins.
    The test will look for the following:
    a) Are the two test tones correctly placed in their respective frequency channels?

    """
    # Specify Channel threshold to decide if energy present is significant
    channel_threshold = 1e5

    # Generate CW to test DDC
    cw_scale = 1
    # Place a tone in the bin center: Lower band edge
    freq1 = 51019287.109375
    # Place a tone in the bin center: Upper band edge
    freq2 = 148980712.890625
    sampling_frequency = 1712e6
    fft_length = np.power(2, 15)
    noise_scale = 0
    mixing_freq = 100e6
    total_samples = np.power(2, 21)

    # Generate the CW for the test: CW for lower band edge
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
    data = cw1 + cw2

    # Run the DDC on test CW
    cpu_decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft
    cpu_decimated_data_trunc = cpu_decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    cpu_ddc_fft = np.power(np.fft.fft(cpu_decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    cpu_above_threshold_channels = np.where(np.abs(cpu_ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 2.
    assert np.size(cpu_above_threshold_channels) == 2

    # Specify expected channel where we expect the translation to occur
    expected_translation_negative_band_edge = len(cpu_decimated_data_trunc) - np.floor(
        (mixing_freq - freq1) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )
    expected_translation_positive_band_edge = np.floor(
        (freq2 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    assert cpu_above_threshold_channels[0][0] == expected_translation_positive_band_edge
    assert cpu_above_threshold_channels[0][1] == expected_translation_negative_band_edge

    """ GPU DDC Compute """
    fir_size = 256

    # A big number. Just be careful not to overload the GPU.
    input_samples = total_samples
    output_samples = input_samples / DDC_fixture.decimation_factor

    # Calculate number of blocks
    samples_per_block = fir_size * 16
    total_blocks = int(input_samples / samples_per_block)
    print(f"Total Blocks is {total_blocks}")

    # I'm using print() statements instead of comments here so that you can see what's happening when you run the script. It can take quite a while to finish...
    print(
        "Setting up memory on host and device for input_data, fir_coeffs (input) and data_downsampled_out (output) vectors"
    )
    data_in_host = cuda.pagelocked_empty(int(input_samples), dtype=np.float32)
    data_in_device = cuda.mem_alloc(data_in_host.nbytes)

    fir_coeffs_host = cuda.pagelocked_empty(int(fir_size), dtype=np.float32)
    fir_coeffs_device = cuda.mem_alloc(fir_coeffs_host.nbytes)

    data_downsampled_out_host = cuda.pagelocked_empty(
        int(output_samples * 2), dtype=np.float32
    )  # *2 for complex values
    data_downsampled_out_device = cuda.mem_alloc(data_downsampled_out_host.nbytes)

    data_debug_real_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
    data_debug_real_out_device = cuda.mem_alloc(data_debug_real_out_host.nbytes)

    data_debug_imag_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
    data_debug_imag_out_device = cuda.mem_alloc(data_debug_imag_out_host.nbytes)

    print("Reading source file and JIT-compiling kernel...")
    module = SourceModule(open("ddc_kernel.cu").read())
    ddc_kernel = module.get_function("kernel_ddc")

    # We're using the [:] here so that the interpreter just updates the values within the numpy ndarray, rather
    # than clobbering the specially-prepared pagelocked memory with a new ndarray.
    fir_coeffs_host[:] = DDC_fixture.ddc_filter_coeffs

    # Compute number of iterations to work through data.
    num_chunks = int(total_samples / input_samples)
    print(f"Num chunks is {num_chunks}")

    gpu_decimated_data = []  # type: List[float]
    start = 0
    end = 0

    for chunk_number in range(num_chunks):
        data_in_host[0:input_samples] = data[
            chunk_number * input_samples : (chunk_number * input_samples + input_samples)
        ]

        print("Copying input data to device...")
        cuda.memcpy_htod(data_in_device, data_in_host)
        cuda.memcpy_htod(fir_coeffs_device, fir_coeffs_host)

        print("Executing kernel...")
        ddc_kernel(
            data_in_device,
            fir_coeffs_device,
            data_downsampled_out_device,
            np.float32(mixing_freq),
            np.int32(chunk_number),
            data_debug_real_out_device,
            data_debug_imag_out_device,  # You can't pass a kernel a normal python int as an argument, it needs to be a numpy dtype.
            block=(fir_size, 1, 1),  # CUDA block and grid sizes can be 3-dimensional,
            grid=(total_blocks, 1, 1),
        )  # but we're just using a 1D vector here.

        print("Copying output data back to the host...")
        cuda.memcpy_dtoh(data_downsampled_out_host, data_downsampled_out_device)
        # cuda.memcpy_dtoh(data_debug_real_out_host, data_debug_real_out_device)
        # cuda.memcpy_dtoh(data_debug_imag_out_host, data_debug_imag_out_device)

        decimated_data = np.array(data_downsampled_out_host[0::2]) + np.array(data_downsampled_out_host[1::2]) * 1j

        start = chunk_number * len(decimated_data)
        end = start + len(decimated_data)
        gpu_decimated_data[start:end] = decimated_data[:]

    # Extract length of data for fft
    gpu_decimated_data_trunc = gpu_decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    gpu_ddc_fft = np.power(np.fft.fft(gpu_decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    gpu_above_threshold_channels = np.where(np.abs(gpu_ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 2.
    assert np.size(gpu_above_threshold_channels) == 2

    # Specify expected channel where we expect the translation to occur
    expected_translation_negative_band_edge = len(gpu_decimated_data_trunc) - np.floor(
        (mixing_freq - freq1) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )
    expected_translation_positive_band_edge = np.floor(
        (freq2 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    assert gpu_above_threshold_channels[0][0] == expected_translation_positive_band_edge
    assert gpu_above_threshold_channels[0][1] == expected_translation_negative_band_edge


def test_run_ddc_out_of_band_cw(DDC_fixture):
    """Test to verify correct exclusion of out-of-band cw tone .

    The purpose of this test is to check the correct exclusion of an out-of-band CW tone.
    This will differ depending on the NarrowBand mode to be tested.

    Test Overview:
    This test will inject two CW tones - one in band and another out-of-band.
    The intention is to check for suitable supression of the out of band tone. The two test
    CW tones are 100MHz (in-band as the mixing frequency is also 100MHz) and 214MHz (out-of-band).
    The out-of-band tone should be greater than 60dB below the in-band tone for the test to pass.
    The test will look for the following:
    a) Is the in-band tone in the correct frequency channel?
    b) Is the out-of-band tone > 60dB below the in-band frequency channel?

    """
    total_samples = 8192 * 8
    input_samples = 8192 * 4
    # Calculate number of blocks
    samples_per_block = 4096

    total_blocks = int(input_samples / samples_per_block)
    print(f"Total Blocks is {total_blocks}")

    # Specify Channel threshold to decide if energy present is significant
    channel_threshold = 1e3

    # Generate CW to test DDC
    cw_scale = 1
    freq1 = 100e6
    freq2 = 214e6
    sampling_frequency = 1712e6
    fft_length = np.power(2, 15)

    num_samples = fft_length * DDC_fixture.decimation_factor * 2
    noise_scale = 0
    mixing_freq = freq1
    signal_to_noise_ratio_threshold_dB = 60.0

    # Generate the CW for the test: CW for band center
    cw1 = cwg.generate_carrier_wave(
        cw_scale=cw_scale,
        freq=freq1,
        sampling_frequency=sampling_frequency,
        num_samples=num_samples,
        noise_scale=noise_scale,
        complex=False,
    )
    # Generate the CW for the test: CW for out of band
    cw2 = cwg.generate_carrier_wave(
        cw_scale=cw_scale,
        freq=freq2,
        sampling_frequency=sampling_frequency,
        num_samples=num_samples,
        noise_scale=noise_scale,
        complex=False,
    )

    # Combine both CW tones prior to putting through the DDC
    data = cw1 + cw2

    # Run the DDC on test CW
    cpu_decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft
    cpu_decimated_data_trunc = cpu_decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    cpu_ddc_fft = np.power(np.fft.fft(cpu_decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    cpu_above_threshold_channels = np.where(np.abs(cpu_ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 1.
    assert np.size(cpu_above_threshold_channels) == 1

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = np.floor(
        (freq1 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    assert cpu_above_threshold_channels == expected_translation_center_channel

    # Get energy in known center channel
    cpu_energy_center_channel = cpu_ddc_fft[cpu_above_threshold_channels]

    # Clear center channel and check next highest channel power.
    cpu_ddc_fft[cpu_above_threshold_channels] = 0

    # Find where the next maximum occurs.
    cpu_next_highest_channel = np.where(np.abs(cpu_ddc_fft) == np.max(np.abs(cpu_ddc_fft)))

    # Get Energy in next highest channel. This should be as a result of the out of band tone.
    cpu_energy_next_highest_channel = cpu_ddc_fft[cpu_next_highest_channel]

    # Compute dB difference. This should be greater than 60dB.
    cpu_dB_diff = 10 * np.log10(np.abs(cpu_energy_center_channel / cpu_energy_next_highest_channel))

    # Check if the test passes
    logging.info(f"dB diff: {cpu_dB_diff}")
    assert cpu_dB_diff > signal_to_noise_ratio_threshold_dB

    """ GPU DDC Compute """
    fir_size = 256

    # A big number. Just be careful not to overload the GPU.
    input_samples = total_samples
    output_samples = input_samples / DDC_fixture.decimation_factor

    # Calculate number of blocks
    samples_per_block = fir_size * 16
    total_blocks = int(input_samples / samples_per_block)
    print(f"Total Blocks is {total_blocks}")

    # I'm using print() statements instead of comments here so that you can see what's happening when you run the script. It can take quite a while to finish...
    print(
        "Setting up memory on host and device for input_data, fir_coeffs (input) and data_downsampled_out (output) vectors"
    )
    data_in_host = cuda.pagelocked_empty(int(input_samples), dtype=np.float32)
    data_in_device = cuda.mem_alloc(data_in_host.nbytes)

    fir_coeffs_host = cuda.pagelocked_empty(int(fir_size), dtype=np.float32)
    fir_coeffs_device = cuda.mem_alloc(fir_coeffs_host.nbytes)

    data_downsampled_out_host = cuda.pagelocked_empty(
        int(output_samples * 2), dtype=np.float32
    )  # *2 for complex values
    data_downsampled_out_device = cuda.mem_alloc(data_downsampled_out_host.nbytes)

    data_debug_real_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
    data_debug_real_out_device = cuda.mem_alloc(data_debug_real_out_host.nbytes)

    data_debug_imag_out_host = cuda.pagelocked_empty(int(input_samples + fir_size), dtype=np.float32)
    data_debug_imag_out_device = cuda.mem_alloc(data_debug_imag_out_host.nbytes)

    print("Reading source file and JIT-compiling kernel...")
    module = SourceModule(open("ddc_kernel.cu").read())
    ddc_kernel = module.get_function("kernel_ddc")

    # We're using the [:] here so that the interpreter just updates the values within the numpy ndarray, rather
    # than clobbering the specially-prepared pagelocked memory with a new ndarray.
    fir_coeffs_host[:] = DDC_fixture.ddc_filter_coeffs

    # Compute number of iterations to work through data.
    num_chunks = int(total_samples / input_samples)
    print(f"Num chunks is {num_chunks}")

    gpu_decimated_data = []  # type: List[float]
    start = 0
    end = 0

    for chunk_number in range(num_chunks):
        data_in_host[0:input_samples] = data[
            chunk_number * input_samples : (chunk_number * input_samples + input_samples)
        ]

        print("Copying input data to device...")
        cuda.memcpy_htod(data_in_device, data_in_host)
        cuda.memcpy_htod(fir_coeffs_device, fir_coeffs_host)

        print("Executing kernel...")
        ddc_kernel(
            data_in_device,
            fir_coeffs_device,
            data_downsampled_out_device,
            np.float32(mixing_freq),
            np.int32(chunk_number),
            data_debug_real_out_device,
            data_debug_imag_out_device,  # You can't pass a kernel a normal python int as an argument, it needs to be a numpy dtype.
            block=(fir_size, 1, 1),  # CUDA block and grid sizes can be 3-dimensional,
            grid=(total_blocks, 1, 1),
        )  # but we're just using a 1D vector here.

        print("Copying output data back to the host...")
        cuda.memcpy_dtoh(data_downsampled_out_host, data_downsampled_out_device)
        # cuda.memcpy_dtoh(data_debug_real_out_host, data_debug_real_out_device)
        # cuda.memcpy_dtoh(data_debug_imag_out_host, data_debug_imag_out_device)

        decimated_data = np.array(data_downsampled_out_host[0::2]) + np.array(data_downsampled_out_host[1::2]) * 1j

        start = chunk_number * len(decimated_data)
        end = start + len(decimated_data)
        gpu_decimated_data[start:end] = decimated_data[:]

    # Extract length of data for fft
    gpu_decimated_data_trunc = gpu_decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    gpu_ddc_fft = np.power(np.fft.fft(gpu_decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    gpu_above_threshold_channels = np.where(np.abs(gpu_ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 1.
    assert np.size(gpu_above_threshold_channels) == 1

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = np.floor(
        (freq1 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    assert gpu_above_threshold_channels == expected_translation_center_channel

    # Get energy in known center channel
    gpu_energy_center_channel = gpu_ddc_fft[gpu_above_threshold_channels]

    # Clear center channel and check next highest channel power.
    gpu_ddc_fft[gpu_above_threshold_channels] = 0

    # Find where the next maximum occurs.
    gpu_next_highest_channel = np.where(np.abs(gpu_ddc_fft) == np.max(np.abs(gpu_ddc_fft)))

    # Get Energy in next highest channel. This should be as a result of the out of band tone.
    gpu_energy_next_highest_channel = gpu_ddc_fft[gpu_next_highest_channel]

    # Compute dB difference. This should be greater than 60dB.
    gpu_dB_diff = 10 * np.log10(np.abs(gpu_energy_center_channel / gpu_energy_next_highest_channel))

    # Check if the test passes
    logging.info(f"dB diff: {gpu_dB_diff}")
    assert gpu_dB_diff > signal_to_noise_ratio_threshold_dB


# def test_run_ddc_center_cw_batch(DDC_fixture):
#     """Test to verify correct translation of center frequency CW down to baseband (DC).

#     The purpose of this test is to check the correct translation of the center frequency CW.

#     Test Overview:
#     This test will inject a single CW tone in the band center.
#     The intention is to check for the correct position of the CW tone in frequency domain.
#     The test CW tone is 100MHz for a band center of 100MHz.
#     The test will look for the following:
#     a) Is the test tone correctly placed in it's respective frequency channel?

#     """
#     # Specify Channel threshold to decide if energy present is significant
#     channel_threshold = 1e5

#     # Generate CW to test DDC
#     cw_scale = 1
#     freq = 100e6
#     sampling_frequency = 1712e6
#     noise_scale = 0
#     mixing_freq = 100e6
#     fft_length = np.power(2, 15)
#     num_samples = fft_length * DDC_fixture.decimation_factor * 2

#     # Generate the CW for the test
#     data = cwg.generate_carrier_wave(
#         cw_scale=cw_scale,
#         freq=freq,
#         sampling_frequency=sampling_frequency,
#         num_samples=num_samples,
#         noise_scale=noise_scale,
#         complex=False,
#     )

#     # Run the DDC on test CW
#     decimated_data = DDC_fixture.run(data, mixing_freq)

#     # Extract length of data for fft. In this test the mixing CW is the same frequency as the test CW.
#     decimated_data_trunc = decimated_data[-fft_length:]

#     # Compute FFT and square to get power spectrum
#     ddc_fft = np.power(np.fft.fft(decimated_data_trunc, axis=-1), 2)

#     # Find where the maximum occurs. This should be the DC bin.
#     above_threshold_channels = np.where(np.abs(ddc_fft) > channel_threshold)

#     # Check if the number of returned channels with energy above the threshold is equal to 1.
#     assert len(above_threshold_channels) == 1

#     # Specify expected channel where we expect the translation to occur. In this test it should be the DC bin(0)
#     expected_translation_center_channel = np.floor(
#         (freq - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
#     )

#     # Check if the test passes
#     ddc_translation_center_channel = above_threshold_channels[0][0]
#     assert ddc_translation_center_channel == expected_translation_center_channel
