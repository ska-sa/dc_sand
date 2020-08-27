import mixer
import pytest
import numpy as np

@pytest.fixture
def Mixer_test():
    mixing_cw = [] 
    input_data = []
    return mixer.Mixer(mixing_cw=mixing_cw, input_data=input_data)

def test_mixer(Mixer_test):
    # Test to verify mixing products
    pass