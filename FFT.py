import numpy as np
from numpy.fft import fft
from typing import Tuple
from config import CONFIG_MIC

def compute_fourier_transform(input_array: np.ndarray, sample_rate = CONFIG_MIC.SAMPLE_RATE.value) -> Tuple[np.ndarray, np.ndarray]:
    sample_interval = 1.0/sample_rate
    t = np.arange(0, 1, sample_interval)
    
    raw_fft = fft(input_array)
    fft_length = len(raw_fft)
    magnitudes = np.abs(raw_fft)
    frequencies = np.arange(fft_length) * (sample_rate / fft_length)

    limit = int(fft_length/2-1)
    return (frequencies[0:limit], magnitudes[0:limit])


