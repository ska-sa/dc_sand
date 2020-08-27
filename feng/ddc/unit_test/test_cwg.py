import cwg
import pytest
import numpy as np

@pytest.fixture
def CWG_test():
    freq = 100e6
    num_samples = 4096
    dither = False
    return cwg.CWG(freq, num_samples, dither)

def test_cw_gen(CWG_test):
    # Test to verify CW generated for mixing CW as well as test vectors
    pass