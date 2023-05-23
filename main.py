import pyaudio
import numpy as np
from PyQt5.QtWidgets import QApplication
from GrapherTool import Plotter
import sys
from config import CONFIG
from FFT import compute_fourier_transform
import time

def convert_seconds_to_frames(seconds:float) -> int:
    rate = CONFIG.SAMPLE_RATE.value
    return int(rate * seconds)

def pyaudio_experiment():
    p = pyaudio.PyAudio()
    stream = p.open(
        format = CONFIG.SAMPLE_TYPE.value,
        channels = CONFIG.CHANNELS.value,
        rate=CONFIG.SAMPLE_RATE.value,
        frames_per_buffer=CONFIG.BUFSIZE.value,
        input=True
    )
    app = QApplication(sys.argv)
    plot = Plotter()

    # to capture a frequency, nyquist theorem says that our sample rate must be twice the 
  
    while True:
        start_time = time.time()
     
        buffer = stream.read(CONFIG.BUFSIZE.value)
        numpy_buffer = np.frombuffer(buffer, dtype=np.int16)

        frequencies, magnitudes = compute_fourier_transform(numpy_buffer)
        plot.update_sample_plot(numpy_buffer)
        plot.update_fourier_plot(frequencies, magnitudes)
        print("FPS:", (int(100*(1.0/(time.time() - start_time))))/100, ", max:", CONFIG.SAMPLE_RATE.value / CONFIG.BUFSIZE.value)
        


if __name__ == "__main__":
    pyaudio_experiment()