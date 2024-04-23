import pyaudio
import numpy as np
from PyQt5.QtWidgets import QApplication
from GrapherTool import Plotter
import sys
from config import CONFIG_MIC
from FFT import compute_fourier_transform
import time
import math
import wave

def convert_seconds_to_frames(seconds:float) -> int:
    rate = CONFIG_MIC.SAMPLE_RATE.value
    return int(rate * seconds)

def pyaudio_experiment1():
    p = pyaudio.PyAudio()
    stream = p.open(
        format = CONFIG_MIC.SAMPLE_TYPE.value,
        channels = CONFIG_MIC.CHANNELS.value,
        rate=CONFIG_MIC.SAMPLE_RATE.value,
        frames_per_buffer=CONFIG_MIC.BUFSIZE.value,
        input=True
    )
    app = QApplication(sys.argv)
    plot = Plotter()

    # to capture a frequency, nyquist theorem says that our sample rate must be twice the 
  
    while True:
        start_time = time.time()
     
        buffer = stream.read(CONFIG_MIC.BUFSIZE.value)
        numpy_buffer = np.frombuffer(buffer, dtype=np.int16)

        frequencies, magnitudes = compute_fourier_transform(numpy_buffer)
       
        plot.update_sample_plot(numpy_buffer)
        plot.update_fourier_plot(frequencies, magnitudes)

def pyaudio_experiment2():
    p = pyaudio.PyAudio()
    chunk = 1024
    wf = wave.open("wav/15b03Lc.wav", 'rb')

    stream = p.open(
        format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(), 
        rate = wf.getframerate(),
        output=True
    )
    print(wf.getframerate())

    data = wf.readframes(chunk)
    app = QApplication(sys.argv)
    plot = Plotter()
    all_data = []
    freq_data = []
    while data:
        data = wf.readframes(chunk)
        all_data.append(data)
    all_data = all_data[0:40]
    
    for data in all_data:
        stream.write(data)
    
    for data in all_data:
        numpy_buffer = np.frombuffer(data, dtype=np.int16)
        if len(numpy_buffer) == 0:
            continue
        frequencies, magnitudes = compute_fourier_transform(numpy_buffer)
        freq_data.append(frequencies)
        plot.update_sample_plot(numpy_buffer)
        plot.update_fourier_plot(frequencies, magnitudes)
    
    samples, sample_length = len(freq_data), len(freq_data[0])

    print(samples, sample_length, samples*sample_length)
        

    wf.close()
    stream.close()
    p.terminate()

def wav_to_emotion_freq_list(wave_file, max_frames):
    
    german_emotion_letter = wave_file[-6]
    emotion = {
        'W': 'anger',
        'L': 'boredom',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happiness',
        'T': 'sadness',
        'N': 'neutral'
    }[german_emotion_letter]
    chunk_size = 1024
    wf = wave.open(wave_file, 'rb')

    data = wf.readframes(chunk_size)
    all_data = []
    while data:
        data = wf.readframes(chunk_size)
        all_data.append(data)
    if len(all_data) < max_frames or len(all_data[max_frames-1]) != len(all_data[0]):
        return None, None
    all_data = all_data[0:max_frames]
    print(len(all_data), len(all_data[0]))
    mag_data = np.array([])
    for data in all_data:
        
        numpy_buffer = np.frombuffer(data, dtype=np.int16)
        if len(numpy_buffer) == 0:
            continue
        frequencies, magnitudes = compute_fourier_transform(numpy_buffer)
        mag_data = np.concatenate((mag_data, magnitudes))

    
    return emotion, mag_data

if __name__ == "__main__":
    emotion, the_list = wav_to_emotion_freq_list("wav/15b03Lc.wav", 25) 
    print(emotion, len(the_list))
    emotion, the_list = wav_to_emotion_freq_list("wav/03a01Wa.wav", 25) 
    print(emotion, len(the_list))