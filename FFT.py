import numpy as np

def compute_fourier_transform(input_array: np.ndarray):
    raw_fft = np.fft.fft(input_array)
    mag_fft = np.abs(raw_fft)
    
    return mag_fft



