import pyaudio
import numpy as np
from PyQt5.QtWidgets import QApplication
from GrapherTool import Plotter
import sys
from config import CONFIG
from FFT import compute_fourier_transform

def convert_seconds_to_frames(seconds:float) -> int:
    rate = CONFIG.RES.value
    return int(rate * seconds)

def pyaudio_experiment(seconds:float) -> None:
    p = pyaudio.PyAudio()
    stream = p.open(
        format = CONFIG.SAMPLE_TYPE.value,
        channels = CONFIG.CHANNELS.value,
        rate=CONFIG.RES.value,
        frames_per_buffer=CONFIG.BUFSIZE.value,
        input=True
    )
    app = QApplication(sys.argv)
    plot = Plotter()

    # to capture a frequency, nyquist theorem says that our sample rate must be twice the 
  
    while True:
        buffer = stream.read(CONFIG.BUFSIZE.value)
        numpy_buffer = np.frombuffer(buffer, dtype=np.int16)
        plot.update_sample_plot(numpy_buffer)
        
        


if __name__ == "__main__":
    pyaudio_experiment(seconds= 0.01)