import decimator
import pytest
import numpy as np

@pytest.fixture
def Decimator():
    decimation_factor = 1
    center_freq = 100e6
    num_samples = 4096
    filter_coeffs = []
    data = []
    return decimator.decimator(decimation_factor=decimation_factor, data=data)

def test_decimate(Decimator):
    # Test to verify decimation. 
    pass