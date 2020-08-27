import unpack_8bit_10bit_data
import pytest
import numpy as np

@pytest.fixture
def Unpack_test():
    data = []
    return unpack_8bit_10bit_data.unpack(data)

def test_unpack_data(Unpack_test):
    # Test to verify CW generated for mixing CW as well as test vectors
    pass