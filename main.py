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
import os
import pandas as pd
import matplotlib.pyplot as plt

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

def play_the_audio(file):
    p = pyaudio.PyAudio()
    chunk = 1024
    wf = wave.open(file, 'rb')

    stream = p.open(
        format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(), 
        rate = wf.getframerate(),
        output=True
    )
    print(wf.getframerate())

    data = wf.readframes(chunk)
    all_data = []
    freq_data = []
    while data:
        data = wf.readframes(chunk)
        all_data.append(data)
    all_data = all_data[0:40]

    if len(all_data) >= 25:
        
        for data in all_data:
            stream.write(data)
    wf.close()
    stream.close()
    p.terminate()


def pyaudio_experiment2():
    p = pyaudio.PyAudio()
    chunk = 1024
    wf = wave.open("wav/03a01Wa.wav", 'rb')

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

def parse_emotion(filename):
    german_emotion_letter = filename[-6]
    return {
        'W': 'anger',
        'L': 'boredom',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happiness',
        'T': 'sadness',
        'N': 'neutral'
    }[german_emotion_letter]

def get_audio_chunks(wave_file, chunk_size):
    wf = wave.open(wave_file, 'rb')
    data = wf.readframes(chunk_size)
    all_data = []
    while data:
        data = wf.readframes(chunk_size)
        all_data.append(data)
    return all_data

def get_framerate(wave_file):
    return wave.open(wave_file, 'rb').getframerate()

def wav_to_emotion_freq_list(wave_file, max_frames):
    emotion = parse_emotion(wave_file)
    
    chunk_size = 1024
    all_data = get_audio_chunks(wave_file, chunk_size)
    
    if len(all_data) < max_frames or len(all_data[max_frames-1]) != len(all_data[0]):
        return None, None, None
    all_data = all_data[0:max_frames]
    mag_data = np.array([])
    labels = list()
    chunk_number = 0
    for data in all_data:
        
        numpy_buffer = np.frombuffer(data, dtype=np.int16)
        if len(numpy_buffer) == 0:
            continue
        frequencies, magnitudes = compute_fourier_transform(numpy_buffer, get_framerate(wave_file))
        new_labels = [f"chunk{chunk_number}_mag{i}_freq{int(frequencies[i])}hz" for i in range(len(magnitudes))]
        chunk_number += 1
        labels +=  new_labels
        mag_data = np.append(mag_data, magnitudes)
    
    return emotion, mag_data, np.array(labels)

def files_iterator(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            yield filepath

def experiment_number_3():
    emotion, the_list, labels = wav_to_emotion_freq_list("wav/15b03Lc.wav", 25) 
    print(emotion, len(the_list))
    emotion, the_list, labels = wav_to_emotion_freq_list("wav/03a01Wa.wav", 25) 
    print(emotion, len(the_list))

def create_dataset():
    none_total = 0
    total  = 0

    per_row_data = []    
    for filename in files_iterator("./wav"):
        emotion, frequency_magnitudes, labels = wav_to_emotion_freq_list(filename, 25)
        our_labels = labels
        if emotion is None:
            none_total += 1
            total += 1
            continue            
        total += 1
        string_array = np.array([filename, emotion])
        concatenated_array = np.concatenate((string_array, frequency_magnitudes))
        per_row_data.append(concatenated_array)
    
    mag_vec_size = len(per_row_data[0])-2
    column_names = ['filename', 'emotion'] + list(our_labels)

    df = pd.DataFrame(data=per_row_data, columns=column_names)
    print(df.head())

    print(f"{none_total} out of {total} were too short")
    return df




def display_file(filename):
    frames = 25
    emotion, frequency_magnitudes, labels = wav_to_emotion_freq_list(filename, 25)
    # Generate some random data for demonstration
    if emotion is None:
        return
    resized = np.resize(frequency_magnitudes, (len(frequency_magnitudes)//frames,frames)) 
    resized = np.sqrt(resized)
    resized = resized.astype(int)
    # Display the image
    plt.imshow(resized, cmap='inferno', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.title(f'{emotion} visualization, file {filename}')
    plt.xlabel('Time Progresses THAT Way =>')
    plt.ylabel('Higher Frequencies THAT Way =>')
    plt.show()

def experiment_number_4():
    for file in files_iterator('./wav'):
        play_the_audio(file)
        display_file(file)

if __name__ == "__main__":
    # pyaudio_experiment1()
    # pyaudio_experiment2()
    # experiment_number_3()
    # create_dataset()
    experiment_number_4()
    
    