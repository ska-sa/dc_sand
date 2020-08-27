import ddc_filter
import pytest
import numpy as np

@pytest.fixture
def DDC_fir_filter_test():
    filter_coeffs = []
    data = []
    return ddc_filter.DDC_fir_Filter(input_data=data, filter_coeffs=filter_coeffs)

def test_fir_filter(DDC_fir_filter_test):
    # Test to verify filter used to remove out-of-band components
    pass