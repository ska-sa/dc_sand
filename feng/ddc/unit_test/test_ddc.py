import ddc
import pytest
import numpy as np

@pytest.fixture
def DDC_test():
    decimation_factor = 1
    center_freq = 100e6
    num_samples = 4096
    filter_coeffs = []
    data = []
    return ddc.DDC(decimation_factor=decimation_factor, center_freq=center_freq, num_samples=num_samples, filter_coeffs=filter_coeffs, data=data)

def test_cw_gen(DDC_test):
    # Test to verify CW generated for mixing CW as well as test vectors
    pass

def test_decimate(DDC_test):
    # Test to verify decimation. 
    pass

def test_decode_8bit_to_10bit_to_float_data(DDC_test):
    # Test to verify conversion from packed 8bit ingest data to 10b (digitiser) data to float
    pass

def test_run_ddc_center_cw(DDC_test):
    # Test to verify correct translation of center frequency CW down to baseband (DC) 
    pass

def test_run_ddc_dual_cw(DDC_test):
    # Test to verify correct translation of center frequecny CW and additional in-band CW
    pass

def test_run_ddc_bandedge_cw(DDC_test):
    # Test to verify correct translation of two in-band CW tones at band edges
    pass

def test_mixer(DDC_test):
    # Test to verify mixing products
    pass

def test_fir_filter():
    # Test to verify filter used to remove out-of-band components
    pass