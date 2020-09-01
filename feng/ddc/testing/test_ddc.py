"""Test Process for Digital Down Conversion."""
import ddc
import pytest


@pytest.fixture
def DDC_fixture():
    """Create DDC test object for pytest."""
    decimation_factor = 1
    filter_coeffs = []
    fs = 1712e6
    return ddc.DigitalDownConverter(decimation_factor=decimation_factor, filter_coeffs=filter_coeffs, fs=fs)


def test_run_ddc_center_cw(DDC_fixture):
    """Test to verify correct translation of center frequency CW down to baseband (DC).

    The purpose of this test is to check the correct translation of the center frequency CW.
    """
    pass


def test_run_ddc_dual_cw(DDC_fixture):
    """Test to verify correct translation of center frequecny CW and additional in-band CW.

    The purpose of this test is to check the correct translation of the center frequency CW as well as
    a second arbitrary CW tone placed mid-band.
    """
    pass


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
