"""Test Process for Digital Down Conversion."""
import ddc
import cwg
import numpy as np
import pytest


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
    """
    # Specify Channel threshold to decide if energy present is significant
    channel_threshold = 1e5

    # Generate CW to test DDC
    cw_scale = 1
    freq = 100e6
    sampling_frequency = 1712e6
    noise_scale = 0
    mixing_freq = freq
    fft_length = np.power(2, 15)
    num_samples = fft_length * DDC_fixture.decimation_factor * 2

    # Generate the CW for the test
    data = cwg.generate_carrier_wave(
        cw_scale=cw_scale,
        freq=freq,
        sampling_frequency=sampling_frequency,
        num_samples=num_samples,
        noise_scale=noise_scale,
        complex=False,
    )

    # Run the DDC on test CW
    decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft. In this test the mixing CW is the same frequency as the test CW.
    decimated_data_trunc = decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    ddc_fft = np.power(np.fft.fft(decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    above_threshold_channels = np.where(np.abs(ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 1.
    assert len(above_threshold_channels) == 1

    # Specify expected channel where we expect the translation to occur. In this test it should be the DC bin(0)
    expected_translation_center_channel = np.floor(
        (freq - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    ddc_translation_center_channel = above_threshold_channels[0][0]
    assert ddc_translation_center_channel == expected_translation_center_channel


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
    sampling_frequency = 1712e6
    fft_length = np.power(2, 15)
    num_samples = fft_length * DDC_fixture.decimation_factor * 2
    noise_scale = 0
    mixing_freq = freq1

    # Generate the CW for the test: CW for band centerde
    cw1 = cwg.generate_carrier_wave(
        cw_scale=cw_scale,
        freq=freq1,
        sampling_frequency=sampling_frequency,
        num_samples=num_samples,
        noise_scale=noise_scale,
        complex=False,
    )
    # Generate the CW for the test: CW for arbitrary tone
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
    decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft
    decimated_data_trunc = decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    ddc_fft = np.power(np.fft.fft(decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    above_threshold_channels = np.where(np.abs(ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 2.
    assert np.size(above_threshold_channels) == 2

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = np.floor(
        (freq1 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )
    expected_translation_off_center_channel = np.floor(
        (freq2 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    assert above_threshold_channels[0][0] == expected_translation_center_channel
    assert above_threshold_channels[0][1] == expected_translation_off_center_channel


def test_run_ddc_bandedge_cw(DDC_fixture):
    """Test to verify correct translation of two in-band CW tones at band edges.

    The purpose of this test is to check the correct translation of the two CW tones placed at the band edges.
    This will differ depending on the NarrowBand mode to be tested.

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
    num_samples = fft_length * DDC_fixture.decimation_factor * 2
    noise_scale = 0
    mixing_freq = 100e6

    # Generate the CW for the test: CW for lower band edge
    cw1 = cwg.generate_carrier_wave(
        cw_scale=cw_scale,
        freq=freq1,
        sampling_frequency=sampling_frequency,
        num_samples=num_samples,
        noise_scale=noise_scale,
        complex=False,
    )
    # Generate the CW for the test: CW for upper band edge
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
    decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft
    decimated_data_trunc = decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    ddc_fft = np.power(np.fft.fft(decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    above_threshold_channels = np.where(np.abs(ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 2.
    assert np.size(above_threshold_channels) == 2

    # Specify expected channel where we expect the translation to occur
    expected_translation_negative_band_edge = len(decimated_data_trunc) - np.floor(
        (mixing_freq - freq1) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )
    expected_translation_positive_band_edge = np.floor(
        (freq2 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    assert above_threshold_channels[0][0] == expected_translation_positive_band_edge
    assert above_threshold_channels[0][1] == expected_translation_negative_band_edge


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
    decimated_data = DDC_fixture.run(data, mixing_freq)

    # Extract length of data for fft
    decimated_data_trunc = decimated_data[-fft_length:]

    # Compute FFT and square to get power spectrum
    ddc_fft = np.power(np.fft.fft(decimated_data_trunc, axis=-1), 2)

    # Find where the maximum occurs. This should be the DC bin.
    above_threshold_channels = np.where(np.abs(ddc_fft) > channel_threshold)

    # Check if the number of returned channels with energy above the threshold is equal to 1.
    assert np.size(above_threshold_channels) == 1

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = np.floor(
        (freq1 - mixing_freq) / ((sampling_frequency / DDC_fixture.decimation_factor) / fft_length)
    )

    # Check if the test passes
    assert above_threshold_channels == expected_translation_center_channel

    # Get energy in known center channel
    Energy_center_channel = ddc_fft[above_threshold_channels]

    # Clear center channel and check next highest channel power.
    ddc_fft[above_threshold_channels] = 0

    # Find where the next maximum occurs.
    next_highest_channel = np.where(np.abs(ddc_fft) == np.max(np.abs(ddc_fft)))

    # Get Energy in next highest channel. This should be as a result of the out of band tone.
    Energy_next_highest_channel = ddc_fft[next_highest_channel]

    # Compute dB difference. This should be greater than 60dB.
    dB_diff = 10 * np.log10(np.abs(Energy_center_channel / Energy_next_highest_channel))

    # Check if the test passes
    assert dB_diff > signal_to_noise_ratio_threshold_dB
