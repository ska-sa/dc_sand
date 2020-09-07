"""Digital Down Conversion Module for FEngine."""
import numpy as np
import cwg
from numpy import genfromtxt
from scipy import signal
import matplotlib.pyplot as plt


class DigitalDownConverter:
    """Digital Down Conversion."""

    def __init__(self, decimation_factor: int, fs: int) -> None:
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
        self.ddc_filter_coeffs = self._import_ddc_filter_coeffs(
            "/home/avanderbyl/Git/dc_sand/feng/ddc/src/ddc_coeff_107MHz.csv"
        )
        self.fs = fs

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
        # filename: str = "/home/avanderbyl/Git/dc_sand/feng/ddc/src/ddc_coeff_107MHz.csv"
        print(f"Importing coefficients from {filename}")
        ddc_coeffs = genfromtxt(filename, delimiter=",")
        print(f"Imported {len(ddc_coeffs)} coefficients")
        return ddc_coeffs

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
        return input_data[0] * mixing_cw[0]

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
        awgn_scale = 0.0
        fs = self.fs
        num_samples = np.size(input_data)
        mixing_cw = cwg.generate_complx_cw(
            cw_scale=cw_scale, freq=center_freq, fs=fs, num_samples=num_samples, awgn_scale=awgn_scale
        )

        # Translate the selected band.
        mix = self._mix(mixing_cw=mixing_cw, input_data=input_data)

        mixing_cw_fft = np.fft.fft(mixing_cw, axis=-1)
        input_data_fft = np.fft.fft(input_data, axis=-1)
        mix_fft = np.fft.fft(mix, axis=-1)

        plt.figure(1)
        plt.semilogy(mixing_cw_fft[0])

        plt.figure(2)
        plt.semilogy(input_data_fft[0])

        plt.figure(3)
        plt.semilogy(mix_fft)

        # Filter the translated band.
        filtered_data = self._bandpass_fir_filter(mix)

        filtered_cw_fft = np.fft.fft(filtered_data, axis=-1)

        plt.figure(4)
        plt.semilogy(filtered_cw_fft)

        # Decimate the filtered band.
        decimated_data = self._decimate(filtered_data)

        decimated_cw_fft = np.fft.fft(decimated_data, axis=-1)

        plt.figure(5)
        plt.semilogy(decimated_cw_fft)

        plt.show()

        return decimated_data


# filename: str = "/home/avanderbyl/Git/dc_sand/feng/ddc/src/ddc_coeff_107MHz.csv"
# print(f"Importing coefficients from {filename}")
# coeffs = []
# with open(filename, mode='r') as coeff_file:
#     ddc_coeffs = csv.reader(coeff_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     for item in ddc_coeffs:
#         coeffs.append(item)
# print(f"Imported {len(coeffs)} coefficients")
