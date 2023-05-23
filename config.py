from enum import Enum
import pyaudio

class CONFIG(Enum):
    BUFSIZE = 4096 # 4 KB
    SAMPLE_TYPE = pyaudio.paInt16 # 16 bit signed integer
    CHANNELS = 1
    # SAMPLE_RATE = 44100 # Hz
    SAMPLE_RATE = 44100