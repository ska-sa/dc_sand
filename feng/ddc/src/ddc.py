"""Digital Down Conversion Module for FEngine."""
import numpy as np
import cwg
from numpy import genfromtxt
from scipy import signal

# import matplotlib.pyplot as plt


class DigitalDownConverter:
    """Digital Down Conversion."""

    def __init__(self, decimation_factor: int, sampling_frequency: int, ddc_coeff_filename: str) -> None:
        """Digital Down Conversion.

        Parameters
        ----------
        decimation_factor: int
            Down-sampling factor for input data array.
        filter_coeffs: np.ndarray of type float
            Filter coefficients for bandpass filtering.
        sampling_frequency: int
            Sampling rate of the digitiser.

        Returns
        -------
        None.
        """
        self.decimation_factor = decimation_factor
        self._import_ddc_filter_coeffs(filename=ddc_coeff_filename)
        self.sampling_frequency = sampling_frequency

    def _import_ddc_filter_coeffs(self, filename: str = "ddc_filter_coeffs_107.csv"):
        """Import Digital Down Converter Filter Coefficients from file.

        Parameters
        ----------
        filename: str
            Digital Down Converter Filter Coefficients filename.
            The default name (if not passed filename): ddc_filter_coeffs_107.csv
        Returns
        -------
        numpy ndarray of filter coefficients: type float.
        """
        print(f"Importing coefficients from {filename}")
        ddc_coeffs = genfromtxt(filename, delimiter=",")
        print(f"Imported {len(ddc_coeffs)} coefficients")
        self.ddc_filter_coeffs = ddc_coeffs
        # return ddc_coeffs

    def _mix(self, mixing_carrier_wave: np.ndarray, input_data: np.ndarray) -> np.ndarray:
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
        return input_data * mixing_carrier_wave

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

    def _bandpass_fir_filter(self, input_data: np.ndarray) -> np.ndarray:
        """Band-pass filter to remove out-of-band products.

        Parameters
        ----------
        input_data: np.ndarray of type float
            Input array of complex-valued samples of vector to be translated (mixed).

        Returns
        -------
        filtered_data: np.ndarray of type float
            Output array of complex-valued filtered data.
        """
        filtered = signal.convolve(input_data, self.ddc_filter_coeffs, mode="valid") / sum(self.ddc_filter_coeffs)

        return filtered

    def _decimate(self, input_data: np.ndarray) -> np.ndarray:
        """Decimate input data by decimation factor.

        Parameters
        ----------
        input_data: np.ndarray of type float
            Input array of complex-valued samples of filtered vector to be decimated.

        Returns
        -------
        decimated_data: np.ndarray of type float
            Output array of complex-valued down-sampled data.

        """
        # Decimate input array. Keep only every 'n' sample where 'n' is the decimation factor.
        return input_data[0 :: self.decimation_factor]

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
            Output array of complex-valued vector of translated, filtered and decimated data.
        """
        # Sanity check the input data.
        if len(input_data) == 0:
            raise ValueError(f"Too few samples in input data. Received {len(input_data)}")

        # Generate the mixing cw tone.
        cw_scale = 1
        noise_scale = 0.0
        sampling_frequency = self.sampling_frequency
        num_samples = np.size(input_data)
        mixing_carrier_wave = cwg.generate_carrier_wave(
            cw_scale=cw_scale,
            freq=center_freq,
            sampling_frequency=sampling_frequency,
            num_samples=num_samples,
            noise_scale=noise_scale,
            complex=True,
        )

        # Translate the selected band.
        mix = self._mix(mixing_carrier_wave=mixing_carrier_wave, input_data=input_data)

        # Filter the translated band.
        filtered_data = self._bandpass_fir_filter(mix)

        # Decimate the filtered band.
        decimated_data = self._decimate(filtered_data)

        # For Debug:
        # mixing_cw_fft = np.power(np.fft.fft(mixing_carrier_wave, axis=-1), 2)
        # input_data_fft = np.power(np.fft.rfft(input_data, axis=-1), 2)
        # mix_fft = np.power(np.fft.fft(mix, axis=-1), 2)
        # filtered_cw_fft = np.power(np.fft.fft(filtered_data, axis=-1), 2)
        # decimated_cw_fft = np.power(np.fft.fft(decimated_data, axis=-1), 2)

        # plt.figure(1)
        # plt.semilogy(input_data_fft)

        # plt.figure(2)
        # plt.semilogy(mixing_cw_fft)

        # plt.figure(3)
        # plt.semilogy(mix_fft)

        # plt.figure(4)
        # plt.semilogy(filtered_cw_fft)

        # plt.figure(5)
        # plt.semilogy(decimated_cw_fft)

        # plt.show()

        return decimated_data
