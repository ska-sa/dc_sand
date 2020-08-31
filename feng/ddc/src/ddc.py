"""Digital Down Conversion Module for FEngine."""
import numpy as np


class DigitalDownConverter:
    """Digital Down Conversion."""

    def __init__(self, decimation_factor: int, filter_coeffs: np.ndarray) -> None:
        """Digital Down Conversion.

        Parameters
        ----------
        decimation_factor: int
            Down-sampling factor for input data array.
        filter_coeffs: np.ndarray of type float
            Filter coefficients for bandpass filtering.

        Returns
        -------
        None.
        """
        self.decimation_factor = decimation_factor
        self.filter_coeffs = filter_coeffs

    def run(self, input_data: np.ndarray, center_freq: float) -> np.ndarray:
        """Digital Down Conversion.

        Parameters
        ----------
        center_freq: float
            Center Frequency of band to be translated.
        input_data: np.ndarray of type float
            Input array of complex-valued samples of vector to be translated (mixed).

        Returns
        -------
        data_mix: np.ndarray of type float
            Output array of complex-valued of vector product.
        """
        pass

    def _mix(self, mixing_cw: np.ndarray, input_data: np.ndarray) -> np.ndarray:
        """Multiply mixing CW with input data.

        Parameters
        ----------
        mixing_cw: np.ndarray of type float
            Input array of complex-valued samples of the mixing vector.
        input_data: np.ndarray of type float
            Input array of complex-valued samples of vector to be translated (mixed).

        Returns
        -------
        data_mix: np.ndarray of type float
            Output array of complex-valued vector product.
        """
        pass

    def _decimate(self, input_data: np.ndarray, decimation_factor: int) -> np.ndarray:
        """Decimate input data by decimation factor.

        Parameters
        ----------
        decimation_factor: int
            Down-sampling factor for input data array.
        input_data: np.ndarray of type float
            Input array of complex-valued samples of filtered vector to be decimated.

        Returns
        -------
        decimated_data: np.ndarray of type float
            Output array of complex-valued down-sampled data.

        """
        pass

    def _bandpass_fir_filter(self, input_data: np.ndarray) -> np.ndarray:
        """Band-pass filter to remove out-of-band products.

        Parameters
        ----------
        filter_coeffs: np.ndarray of type float
            Filter coefficients for bandpass filtering.
        input_data: np.ndarray of type float
            Input array of complex-valued samples of vector to be translated (mixed).

        Returns
        -------
        filtered_data: np.ndarray of type float
            Output array of complex-valued filtered data.

        """
        pass

    def _decode_8bit_to_10bit_to_float_data(self, data_8bit: np.ndarray) -> np.ndarray:
        """Convert 8bit packed data to 10bit to float.

        Note: Digitiser raw data 10bit and packed into 8bit words for transport.

        Parameters
        ----------
        data_8bit: np.ndarray of type float
            Array of input (8bit-packed) samples to be unpacked.

        Returns
        -------
        decoded_data: np.ndarray of type float
            Array of real-valued unpacked data.
        """
        pass
