"""Test Process for CW Generator."""
import pytest
import numpy as np


@pytest.fixture
def cw_fixture():
    """Fixture for Carrier Wave Generator Test."""
    import cwg

    return cwg


def test_cw_real(cw_fixture):
    """Test to verify real-valued CW generated for mixing CW as well as test vectors."""
    cw_scale = 1
    freq = 100e6
    fs = 1712e6
    num_samples = 8192
    awgn_scale = 0
    cw = cw_fixture.generate_real_cw(
        cw_scale=cw_scale, freq=freq, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale
    )

    # Check if the generated CW is the expected frequency.
    channel_resolution = fs / num_samples
    expected_channel = np.floor(freq / channel_resolution)

    # Compute real FFT on generated CW
    cw_fft = np.fft.rfft(np.real(cw), axis=-1)
    cw_channel = np.where(cw_fft == np.max(cw_fft))

    assert cw_channel[0][0] == expected_channel


def test_cw_complex(cw_fixture):
    """Test to verify complex valued CW generated for mixing CW as well as test vectors."""
    cw_scale = 1
    freq = 214e6
    fs = 1712e6
    num_samples = 8192
    awgn_scale = 0
    cw = cw_fixture.generate_complx_cw(
        cw_scale=cw_scale, freq=freq, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale
    )

    # Check if the generated CW is the expected frequency.
    channel_resolution = fs / num_samples
    expected_channel = np.floor(freq / channel_resolution)
    cw_fft_cmplx = np.fft.fft(cw, axis=-1)

    # Compute Complex FFT on generated CW
    cw_channel = np.where(cw_fft_cmplx == np.max(cw_fft_cmplx))

    assert cw_channel[0][0] == expected_channel
