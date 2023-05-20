import pyaudio
from enum import Enum
import numpy as np
from PyQt5.QtWidgets import QApplication
from GrapherTool import Plotter
import sys

class Record(Enum):
    BUFSIZE = 4096 # 4 KB
    SAMPLE_TYPE = pyaudio.paInt16 # 16 bit signed integer
    CHANNELS = 1
    RES = 44100 # Hz


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
    app = QApplication(sys.argv)
    plot = Plotter()

    while True:
        buffer = stream.read(Record.BUFSIZE.value)
        numpy_buffer = np.frombuffer(buffer, dtype=np.int16)
        plot.update_plot(numpy_buffer)


if __name__ == "__main__":
    pyaudio_experiment(seconds= 0.01)