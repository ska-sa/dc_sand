"""Test Process for CW Generator."""
import cwg
import pytest


@pytest.fixture
def test_cw_real():
    """Test to verify real-valued CW generated for mixing CW as well as test vectors."""
    cw_scale = 1
    freq = 100e6
    fs = 1712e6
    num_samples = 4096
    awgn_scale = 0
    cwg.generate_real_cw(cw_scale=cw_scale, freq=freq, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale)


def test_cw_complex():
    """Test to verify complex valued CW generated for mixing CW as well as test vectors."""
    pass
