"""Carrier Wave (CW) Generator for FEngine."""
import numpy as np
import scipy.stats


def generate_real_cw(cw_scale: float, freq: float, fs: int, num_samples: int, awgn_scale: float) -> np.ndarray:
    """Generate a real-valued CW vector.

    Parameters
    ----------
    cw_scale: float
        factor to scale generated noise.
    freq: float
        Frequency of CW to be generated.
    fs: int
        Sample rate for generated CW.
    num_samples: int
        Number of samples for generated CW.
    awgn_scale: float
        factor to scale generated noise.

    Returns
    -------
    np.ndarray of type float
        Output array of real-valued samples for generated CW.
    """
    samples_per_cycle = fs / freq
    cycles = int((num_samples) / samples_per_cycle)
    in_array = np.linspace(0, (cycles), num_samples)

    # Generate CWG
    cwg = np.real(cw_scale * np.exp(-1j * 2 * np.pi * in_array).astype(np.complex64)).astype(np.float32)

    # Generate AWGN
    awgn = _generate_awgn(awgn_scale, len(cwg))

    return [cwg + awgn]


def generate_complx_cw(cw_scale: float, freq: float, fs: int, num_samples: int, awgn_scale: float) -> np.ndarray:
    """Generate a complex-valued CW vector.

    Parameterž
    ----------
    cw_scale: float
        factor to scale generated noise.
    freq: float
        Frequency of CW to be generated.
    fs: int
        Sample rate for generated CW.
    num_samples: int
        Number of samples for generated CW.
    awgn_scale: float
        factor to scale generated noise.

    Returns
    -------
    np.ndarray of type float
        Output array of complex-valued samples for generated CW.
    """
    samples_per_cycle = fs / freq
    cycles = int((num_samples) / samples_per_cycle)
    in_array = np.linspace(0, (cycles), num_samples)

    # Generate CWG
    cwg = cw_scale * (np.exp(-1j * 2 * np.pi * in_array)).astype(np.complex64)

    # Generate AWGN
    awgn = _generate_awgn(awgn_scale, len(cwg))

    return [cwg + awgn]


def _generate_awgn(scale: float, array_length: int) -> np.ndarray:
    """Generate additive white gaussian noise.

    Parameters
    ----------
    array_length: int
        Number of noise samples to be created.
    scale: float
        factor to scale generated noise.

    Returns
    -------
    np.ndarray of type float
        Array of noise samples.
    """
    lower = -1.0
    upper = 1.0
    mu = 0.0
    sigma = 0.50
    N = array_length

    return scale * scipy.stats.truncnorm.rvs(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N
    ).astype(np.float32)
