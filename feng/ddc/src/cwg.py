"""Carrier Wave (CW) Generator for FEngine."""
import numpy as np
import scipy.stats


def generate_carrier_wave(
    cw_scale: float, freq: float, sampling_frequency: int, num_samples: int, noise_scale: float, complex: bool
) -> np.ndarray:
    """Generate a carrier wave vector.

    Parameters
    ----------
    cw_scale: float
        factor to scale generated noise.
    freq: float
        Frequency of CW to be generated.
    sampling_frequency: int
        Sample rate for generated CW. This is expressed in Hz. E.g. 1712e6.
    num_samples: int
        Number of samples for generated CW.
    noise_scale: float
        Factor to scale generated noise.
    complex: bool
        Specify if realor complex carrier wave is required.

    Returns
    -------
    np.ndarray of type float
        Output array of complex-valued samples for generated CW.
    """
    samples_per_cycle = sampling_frequency / freq
    cycles = int((num_samples) / samples_per_cycle)
    in_array = np.linspace(0, (cycles), num_samples)

    # Generate Carrier Wave.
    carrier_wave_complex = cw_scale * (np.exp(-1j * 2 * np.pi * in_array)).astype(np.complex64)

    # Generate Additive White Gaussian Noise.
    additive_white_gaussian_noise = _generate_noise(noise_scale, len(carrier_wave_complex))

    if complex is True:
        return carrier_wave_complex + additive_white_gaussian_noise
    else:
        return np.real(carrier_wave_complex + additive_white_gaussian_noise)


def _generate_noise(scale: float, array_length: int) -> np.ndarray:
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
