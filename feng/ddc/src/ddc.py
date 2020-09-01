"""Digital Down Conversion Module for FEngine."""
import numpy as np
import cwg


class DigitalDownConverter:
    """Digital Down Conversion."""

    def __init__(self, decimation_factor: int, filter_coeffs: np.ndarray, fs: int) -> None:
        """Digital Down Conversion.

        Parameters
        ----------
        decimation_factor: int
            Down-sampling factor for input data array.
        filter_coeffs: np.ndarray of type float
            Filter coefficients for bandpass filtering.
        fs: int
            Sampling rate of the digitiser.

        Returns
        -------
        None.
        """
        self.decimation_factor = decimation_factor
        self.filter_coeffs = filter_coeffs
        self.fs = fs

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
        np.ndarray of type float
            Output array of complex-valued vector product.
        """
        return input_data * mixing_cw

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
        # Sanity check the input data.
        if len(input_data) == 0:
            raise ValueError(f"Too few samples in input data. Received {len(input_data)}")

        # Generate the mixing cw tone.
        cw_scale = 1
        awgn_scale = 0.0001
        fs = self.fs
        num_samples = len(input_data)
        mixing_cw = cwg.generate_complx_cw(
            cw_scale=cw_scale, freq=center_freq, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale
        )

        # Translate the selected band.
        mix = self._mix(mixing_cw=mixing_cw, input_data=input_data)

        # mixing_cw_fft = np.fft.fft(mixing_cw, axis=-1)
        # input_data_fft = np.fft.fft(input_data, axis=-1)
        # mix_fft = np.fft.fft(mix, axis=-1)

        # plt.figure(1)
        # plt.semilogy(mixing_cw_fft)

        # plt.figure(2)
        # plt.semilogy(input_data_fft)

        # plt.figure(3)
        # plt.semilogy(mix_fft)

        # plt.show()

        # Filter the translated band.

        # Decimate the filtered band.
        return mix

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
        # taps = 1
        # channels = int(len(input_data))
        # samples = 2 * channels * (taps - 1)

        # weights = _generate_weights(channels, taps)
        # expected_fir = _pfb_fir_host(input_data, channels, weights)

        # #np.testing.assert_allclose(h_out, expected_fir, rtol=1e-5, atol=1e-3)

        # # plt.figure(2)
        # # plt.plot(expected[0][0:100])
        # # plt.show()

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
