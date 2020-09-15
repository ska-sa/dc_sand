"""Test Process for CW Generator."""
import pytest
import numpy as np
import logging


@pytest.fixture
def cw_fixture():
    """Fixture for Carrier Wave Generator Test."""
    import cwg

    return cwg


def test_cw_real(cw_fixture):
    """Test to verify real-valued CW generated for mixing CW as well as test vectors."""
    cw_scale = 1
    freq = 100e6
    sampling_frequency = 1712e6
    num_samples = 8192
    noise_scale = 0
    cw = cw_fixture.generate_carrier_wave(
        cw_scale=cw_scale,
        freq=freq,
        sampling_frequency=sampling_frequency,
        num_samples=num_samples,
        noise_scale=noise_scale,
        complex=False,
    )

    # Check if the generated CW is the expected frequency.
    channel_resolution = sampling_frequency / num_samples
    expected_channel = np.floor(freq / channel_resolution)

    # Compute real FFT on generated CW
    cw_fft = np.fft.rfft(np.real(cw), axis=-1)
    cw_channel = np.where(cw_fft == np.max(cw_fft))[0][0]

    logging.info(f"Found cw signal in channel {cw_channel}")
    assert cw_channel == expected_channel


def test_cw_complex(cw_fixture):
    """Test to verify complex valued CW generated for mixing CW as well as test vectors."""
    cw_scale = 1
    freq = 214e6
    sampling_frequency = 1712e6
    num_samples = 8192
    fft_length = 8192
    noise_scale = 0
    cw = cw_fixture.generate_carrier_wave(
        cw_scale=cw_scale,
        freq=freq,
        sampling_frequency=sampling_frequency,
        num_samples=num_samples,
        noise_scale=noise_scale,
        complex=True,
    )

    # Check if the generated CW is the expected frequency.
    channel_resolution = sampling_frequency / num_samples
    expected_channel = np.floor(freq / channel_resolution)

    # Since it is complex (e^(-j)), the tone falls into the negative part of the spectrum.
    # Compute where it would be in the negative part of the spectrum.
    expected_channel = fft_length - expected_channel

    cw_fft_cmplx = np.fft.fft(cw, axis=-1)

    # Compute Complex FFT on generated CW
    cw_channel = np.where(cw_fft_cmplx == np.max(cw_fft_cmplx))[0][0]

    logging.info(f"Found cw signal in channel {cw_channel}")
    assert cw_channel == expected_channel
