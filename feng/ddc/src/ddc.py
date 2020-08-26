import numpy as np
import matplotlib.pyplot as plt

class DDC:
    def __init__(self, decimation_factor: int, center_freq: float, data: np.ndarray, filter_coeffs: np.ndarray) -> None:
        self.decimation_factor = decimation_factor
        self.center_freq = center_freq
        self.filter_coeffs = filter_coeffs
        self.data = data

    def generate_cwg(self, freq: float, dither: bool):
        pass

    def decode_8bit_to_10bit_to_float_data(self, data_8bit: np.ndarray):
        pass

    def decimate(self, data: np.ndarray, decimation_factor: int):
        pass

    def run_ddc(self, osc: np.ndarray, data: np.ndarray, filter_coeffs: np.ndarray, decimation_factor: int):
        pass
