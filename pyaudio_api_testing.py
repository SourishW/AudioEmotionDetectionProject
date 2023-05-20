import pyaudio
import wave 
from enum import Enum

class Record(Enum):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100 # frames per second

def convert_seconds_to_frames(seconds:float) -> int:
    rate = Record.RATE.value
    return int(rate * seconds)

def pyaudio_experiment(seconds:float) -> None:
    p = pyaudio.PyAudio()
    stream = p.open(
        format = Record.FORMAT.value,
        channels = Record.CHANNELS.value,
        rate=Record.RATE.value,
        frames_per_buffer=Record.CHUNK.value,
        input=True
    )
    print(convert_seconds_to_frames(seconds),"frames recorded")
    print(stream.read(convert_seconds_to_frames(seconds)))

if __name__ == "__main__":
    pyaudio_experiment(seconds= 0.01)