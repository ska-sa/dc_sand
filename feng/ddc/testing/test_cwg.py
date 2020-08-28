"""Test Process for CW Generator."""
import cwg
import pytest


@pytest.fixture
def CWG_test():
    """Create CW generator object for pytest."""
    return cwg.CWGenerator()


def test_cw_real(CWG_test):
    """Test to verify real-valued CW generated for mixing CW as well as test vectors."""
    pass


def test_cw_complex(CWG_test):
    """Test to verify complex valued CW generated for mixing CW as well as test vectors."""
    pass
