"""Test Process for Digital Down Conversion."""
import ddc
import cwg
import numpy as np
import pytest


@pytest.fixture
def DDC_fixture():
    """Create DDC test object for pytest."""
    decimation_factor = 8
    fs = 1712e6
    return ddc.DigitalDownConverter(decimation_factor=decimation_factor, fs=fs)


def test_run_ddc_center_cw(DDC_fixture):
    """Test to verify correct translation of center frequency CW down to baseband (DC).

    The purpose of this test is to check the correct translation of the center frequency CW.
    """
    # Generate CW to test DDC
    cw_scale = 1
    freq = 100e6
    fs = 1712e6
    num_samples = 8192
    awgn_scale = 0
    mixing_freq = freq
    data = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)
    decimated_data = DDC_fixture.run(data, mixing_freq)

    ddc_fft = np.fft.rfft(np.real(decimated_data), axis=-1)
    ddc_channel = np.where(ddc_fft == np.max(ddc_fft))

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = 0

    if np.size(ddc_channel) > 1:
        raise ValueError(
            f"Too many channels with energy. Expected only 1 (DC bin). Instead received {len(ddc_channel)} channel(s)."
        )

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
    freq2 = 110e6
    fs = 1712e6
    num_samples = 8192
    awgn_scale = 0
    mixing_freq = freq1

    cw1 = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq1, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)
    cw2 = cwg.generate_real_cw(cw_scale=cw_scale, freq=freq2, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)
    data = [cw1[0] + cw2[0]]

    decimated_data = DDC_fixture.run(data, mixing_freq)

    ddc_fft = np.power(np.fft.fft(decimated_data, axis=-1), 2)
    ddc_channel = np.where(ddc_fft > channel_threshold)

    # Specify expected channel where we expect the translation to occur
    expected_translation_center_channel = 0
    expected_translation_off_center_channel = 47

    if np.size(ddc_channel) > 2:
        raise ValueError(
            f"Too many channels with energy. Expected 2 (DC bin and bin 47 with freq {(freq2-freq1)/1e6}MHz). Instead received {len(ddc_channel)} channel(s)."
        )

    assert ddc_channel[0][0] == expected_translation_center_channel
    assert ddc_channel[0][1] == expected_translation_off_center_channel


def test_run_ddc_bandedge_cw(DDC_fixture):
    """Test to verify correct translation of two in-band CW tones at band edges.

    The purpose of this test is to check the correct translation of the two CW tones placed at the band edges.
    This will differ depending on the NarrowBand mode to be tested.

    """
    pass


def test_run_ddc_out_of_band_cw(DDC_fixture):
    """Test to verify correct exclusion of out-of-band cw tone .

    The purpose of this test is to check the correct exclusion of an out-of-band CW tone.
    This will differ depending on the NarrowBand mode to be tested.

    """
    pass
