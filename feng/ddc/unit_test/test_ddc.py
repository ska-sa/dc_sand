import ddc
import pytest
import numpy as np

@pytest.fixture
def DDC_test():
    decimation_factor = 1
    center_freq = 100e6
    filter_coeffs = [0,0,0,0,0,0,0,0]
    data = [0,0,0,0,0,0,0,0]
    return ddc.DDC(decimation_factor=decimation_factor, center_freq=center_freq, filter_coeffs=filter_coeffs, data=data)

def test_cwg_gen(DDC_test):
    pass

def test_decimate(DDC_test):
    pass

def test_decode_8bit_to_10bit_to_float_data(DDC_test):
    pass

def test_run_ddc(DDC_test):
    pass