import pyaudio
import wave 
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

class Record(Enum):
    BUFSIZE = 4096 # 4 KB
    SAMPLE_TYPE = pyaudio.paInt16 # 16 bit signed integer
    CHANNELS = 1
    RES = 44100 # Hz

class Plotter:
    def __init__(self):
        fig, ax = plt.subplots(1, figsize=(20, 20))
        x = np.arange(0, Record.BUFSIZE.value)
        line, = ax.plot(x, x, lw=2)

        ax.set_title("16 bit signed real time audio signal")
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Time")
        
        # 16 bit signed integers go from -2^15 -1 to 2^15
        low_end = -2**15-1
        high_end = 2**15
        ax.set_ylim(low_end, high_end) 
        plt.setp(ax, yticks = [low_end, low_end/2, 0, high_end/2, high_end])
        plt.show(block = False)

        self.__fig = fig
        self.__line = line

    def update_plot(self, buffer):
        self.__line.set_ydata(buffer)
        self.__fig.canvas.draw()
        self.__fig.canvas.flush_events()

def convert_seconds_to_frames(seconds:float) -> int:
    rate = Record.RES.value
    return int(rate * seconds)

def pyaudio_experiment(seconds:float) -> None:
    p = pyaudio.PyAudio()
    stream = p.open(
        format = Record.SAMPLE_TYPE.value,
        channels = Record.CHANNELS.value,
        rate=Record.RES.value,
        frames_per_buffer=Record.BUFSIZE.value,
        input=True
    )

    plot = Plotter()

    while True:
        buffer = stream.read(Record.BUFSIZE.value)
        numpy_buffer = np.frombuffer(buffer, dtype=np.int16)
        plot.update_plot(numpy_buffer)


if __name__ == "__main__":
    pyaudio_experiment(seconds= 0.01)