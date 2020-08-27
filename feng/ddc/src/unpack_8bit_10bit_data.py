import numpy as np

class unpack:
    def __init__(self, data_8bit: np.ndarray):
        self.data_8bit = data_8bit

    def decode_8bit_to_10bit_to_float_data(self, data_8bit: np.ndarray):
        # Method to convert 8bit packed data to 10bit (initial raw) to float
        pass