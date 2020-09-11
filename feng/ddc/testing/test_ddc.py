"""Test Process for Digital Down Conversion."""
import ddc
import cwg
import numpy as np
import pytest
import logging


@pytest.fixture
def DDC_fixture():
    """Create DDC test object for pytest."""
    decimation_factor = 16
    fs = 1712e6
    ddc_coeff_filename = "../src/ddc_coeff_107MHz.csv"
    return ddc.DigitalDownConverter(decimation_factor=decimation_factor, fs=fs, ddc_coeff_filename=ddc_coeff_filename)


def test_run_ddc_center_cw(DDC_fixture):
    """Test to verify correct translation of center frequency CW down to baseband (DC).

    The purpose of this test is to check the correct translation of the center frequency CW.
    """
    # Specify Channel threshold to decide if energy present is significant
    channel_threshold = 1e5

    # Generate CW to test DDC
    cw_scale = 1
    freq = 100e6
    fs = 1712e6
    num_samples = 8192 * 2
    awgn_scale = 0
    mixing_freq = freq

    # Generate the CW for the test
    data = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)

    # Run the DDC on test CW
    decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft. In this test the mixing CW is the same frequency as the test CW.
    decimated_data_trunc = decimated_data[-1024:]

    # Compute FFT and square to get power spectrum
    ddc_fft = np.power(np.fft.fft(decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    ddc_channel = np.where(np.abs(ddc_fft) > channel_threshold)

    # Specify expected channel where we expect the translation to occur. In this test it should be the DC bin(0)
    expected_translation_center_channel = 0

    # Check if the number of returned channels with energy above the threshold is greater than 1.
    if np.size(ddc_channel) > 1:
        raise ValueError(
            f"Too many channels with energy. Expected only 1 (DC bin). Instead received {len(ddc_channel)} channel(s)."
        )

    # Check if the test passes
    assert ddc_channel[0][0] == expected_translation_center_channel


def test_run_ddc_dual_cw(DDC_fixture):
    """Test to verify correct translation of center frequecny CW and additional in-band CW.

    The purpose of this test is to check the correct translation of the center frequency CW as well as
    a second arbitrary CW tone placed mid-band.
    """
    # Specify Channel threshold to decide if energy present is significant
    channel_threshold = 1e5

    # Generate CW to test DDC
    cw_scale = 1
    freq1 = 100e6
    freq2 = 103343750
    fs = 1712e6
    num_samples = 8192 * 3
    awgn_scale = 0
    mixing_freq = freq1

    # Generate the CW for the test: CW for band center
    cw1 = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq1, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)
    # Generate the CW for the test: CW for arbitrary tone
    cw2 = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq2, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)
    data = [cw1[0] + cw2[0]]

    # Run the DDC on test CW
    decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft
    decimated_data_trunc = decimated_data[-1024:]

    # Compute FFT and square to get power spectrum
    ddc_fft = np.power(np.fft.fft(decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    ddc_channel = np.where(np.abs(ddc_fft) > channel_threshold)

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = 0
    expected_translation_off_center_channel = 32

    # Check if the number of returned channels with energy above the threshold is greater than 2.
    if np.size(ddc_channel) > 2:
        raise ValueError(
            f"Too many channels with energy. Expected 2 (DC bin and bin 47 with freq {(freq2-freq1)/1e6}MHz). Instead received {len(ddc_channel)} channel(s)."
        )

    # Check if the test passes
    assert ddc_channel[0][0] == expected_translation_center_channel
    assert ddc_channel[0][1] == expected_translation_off_center_channel


def test_run_ddc_bandedge_cw(DDC_fixture):
    """Test to verify correct translation of two in-band CW tones at band edges.

    The purpose of this test is to check the correct translation of the two CW tones placed at the band edges.
    This will differ depending on the NarrowBand mode to be tested.

    """
    # Specify Channel threshold to decide if energy present is significant
    channel_threshold = 1e3

    # Generate CW to test DDC
    cw_scale = 1
    freq1 = 47.5e6
    freq2 = 152.5e6
    fs = 1712e6
    num_samples = 8192 * 3
    awgn_scale = 0
    mixing_freq = 100e6

    # Generate the CW for the test: CW for lower band edge
    cw1 = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq1, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)
    # Generate the CW for the test: CW for upper band edge
    cw2 = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq2, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)
    data = [cw1[0] + cw2[0]]

    # Run the DDC on test CW
    decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft
    decimated_data_trunc = decimated_data[-1024:]

    # Compute FFT and square to get power spectrum
    ddc_fft = np.power(np.fft.fft(decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    ddc_channel = np.where(np.abs(ddc_fft) > channel_threshold)

    # Specify expected channel where we expect the translation to occur
    expected_translation_positive_band_edge = 503
    expected_translation_negative_band_edge = len(decimated_data_trunc) - 503

    # Check if the number of returned channels with energy above the threshold is greater than 2.
    if np.size(ddc_channel) > 2:
        raise ValueError(
            f"Too many channels with energy. Expected 2 (DC bin and bin 47 with freq {(freq2-freq1)/1e6}MHz). Instead received {len(ddc_channel)} channel(s)."
        )

    # Check if the test passes
    assert ddc_channel[0][0] == expected_translation_positive_band_edge
    assert ddc_channel[0][1] == expected_translation_negative_band_edge


def test_run_ddc_out_of_band_cw(DDC_fixture):
    """Test to verify correct exclusion of out-of-band cw tone .

    The purpose of this test is to check the correct exclusion of an out-of-band CW tone.
    This will differ depending on the NarrowBand mode to be tested.

    """
    # Specify Channel threshold to decide if energy present is significant
    channel_threshold = 1e3

    # Generate CW to test DDC
    cw_scale = 1
    freq1 = 100e6
    freq2 = 214e6
    fs = 1712e6
    num_samples = 8192 * 3
    awgn_scale = 0
    mixing_freq = freq1

    # Generate the CW for the test: CW for band center
    cw1 = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq1, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)
    # Generate the CW for the test: CW for out of band
    cw2 = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq2, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)
    data = [cw1[0] + cw2[0]]

    # Run the DDC on test CW
    decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft
    decimated_data_trunc = decimated_data[-1024:]

    # Compute FFT and square to get power spectrum
    ddc_fft = np.power(np.fft.fft(decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    ddc_channel = np.where(np.abs(ddc_fft) > channel_threshold)

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = 0

    # Check if the number of returned channels with energy above the threshold is greater than 2.
    if np.size(ddc_channel) > 2:
        raise ValueError(
            f"Too many channels with energy. Expected only 1 (DC bin). Instead received {len(ddc_channel)} channel(s)."
        )

    # Check if the test passes
    assert ddc_channel[0][0] == expected_translation_center_channel

    # Get energy in known center channel
    Energy_center_channel = ddc_fft[0]

    # Clear center channel and check next highest channel power.
    ddc_fft[0] = 0

    # Find where the next maximum occurs.
    ddc_channel = np.where(np.abs(ddc_fft) == np.max(np.abs(ddc_fft)))

    # Get Energy in next highest channel. This should be as a result of the out of band tone.
    Energy_next_highest_channel = ddc_fft[ddc_channel[0][0]]

    # Compute dB difference. This should be greater than 60dB.
    dB_diff = 10 * np.log10(np.abs(Energy_center_channel / Energy_next_highest_channel))
    logging.info(f"dB diff: {dB_diff}")

    # Check if the test passes
    assert dB_diff > 60
