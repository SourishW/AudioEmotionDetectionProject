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
import librosa
import librosa.display
from scipy.io import wavfile

FRAME_NUMBER = 12

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
    print(f"Playing {file}")

    data = wf.readframes(chunk)
    all_data = []
    freq_data = []
    while data:
        data = wf.readframes(chunk)
        all_data.append(data)
    all_data = all_data[0:40]

    if len(all_data) >= FRAME_NUMBER:
        
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
    
    chunk_size = 2048
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
    emotion, the_list, labels = wav_to_emotion_freq_list("wav/15b03Lc.wav", FRAME_NUMBER) 
    print(emotion, len(the_list))
    emotion, the_list, labels = wav_to_emotion_freq_list("wav/03a01Wa.wav", FRAME_NUMBER) 
    print(emotion, len(the_list))

def create_dataset():
    none_total = 0
    total  = 0

    per_row_data = []    
    for filename in files_iterator("./wav"):
        emotion, frequency_magnitudes, labels = wav_to_emotion_freq_list(filename, FRAME_NUMBER)
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


def plot_distribution(data):

    # Plot the histogram
    plt.hist(data, bins=30, density=True, alpha=0.7, color='blue') # adjust the number of bins as needed
    plt.title('Distribution of Numbers')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

def display_file(filename):
    frames = FRAME_NUMBER
    emotion, frequency_magnitudes, labels = wav_to_emotion_freq_list(filename, FRAME_NUMBER)
    # Generate some random data for demonstration
    if emotion is None:
        return
    # plot_distribution(np.sqrt(frequency_magnitudes))
    dimensions = (len(frequency_magnitudes)//frames,frames)
    resized = np.resize(frequency_magnitudes, dimensions ) 
    resized = np.sqrt(resized)
    resized = resized.astype(int)
    # Display the image
    plt.imshow(resized, cmap='inferno', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.title(f'{emotion} visualization, file {filename}')
    y_ticks = np.arange(0, dimensions[0], dimensions[0]//30)

    y_labels = [labels[i][labels[i].find("freq")+4:] for i in y_ticks]  # Get labels corresponding to every 10th frame
    plt.yticks(y_ticks, y_labels)
    plt.xlabel('Time Progresses THAT Way =>')
    plt.ylabel(' <= Higher Frequencies THAT Way')
    plt.show()

def experiment_number_4():
    for file in files_iterator('./wav'):
        play_the_audio(file)
        display_file(file)

def display_mel_spectogram():
    pass

def get_all_audio_samples(wave_file):
    p = pyaudio.PyAudio()
    chunk = 1024
    wf = wave.open("wav/03a01Wa.wav", 'rb')
    print(wf.getframerate())
    data = wf.readframes(wf.getnframes())
    wf.close()
    return data

def librosa_plot(samples, sample_rate):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(samples, sr=sample_rate)
    plt.show()

def experiment_number_5_librosa():
    for file in files_iterator('./wav'):
        play_the_audio(file)
        samples, sample_rate = librosa.load(file, sr=None)
        sgram = librosa.stft(samples)
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        sgram_mag, _ = librosa.magphase(sgram)
        
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        print(mel_sgram.size)
        print(len(mel_sgram), len(mel_sgram[0]))
        img = librosa.display.specshow(mel_sgram,y_axis='linear', x_axis='time', ax = ax)
        print(img)
        ax.set_title(f"{file}, {parse_emotion(file)}")
        
        plt.show()

def generate_image_files(pad_len):
    destination_dir = './img_unsized'
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    
    for file in files_iterator('./wav'):
        
        samples, sample_rate = librosa.load(file, sr=None)
        '''
        if len(samples) > pad_len:
            diff = int(len(samples) - pad_len) + 1
            samples = samples[int(diff / 2) : len(samples) - int(diff / 2)]

        samples = librosa.util.pad_center(samples, size=pad_len)
        '''

        sgram = librosa.stft(samples)
        
        
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        sgram_mag, _ = librosa.magphase(sgram)
        
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        img = librosa.display.specshow(mel_sgram,y_axis='linear', x_axis='time', ax = ax)
        plt.axis('off')

        file_name = file[6:-4] + '.png'
        print(len(mel_sgram), len(mel_sgram[0]))
        print("Writing file:", file_name)

        directory = destination_dir + '/' + parse_emotion(file_name) + '/'
        if not os.path.exists(directory):
            os.mkdir(directory)

        plt.savefig(directory + file_name, bbox_inches='tight', pad_inches=0)
        
        

def get_average_file_len():
    average = 0
    num = 0
    for file in files_iterator('./wav'):
        
        samples, sample_rate = librosa.load(file, sr=None)

        average += len(samples)
        num += 1
    return int(average / num)
        

if __name__ == "__main__":
    # pyaudio_experiment1()
    # pyaudio_experiment2()
    # experiment_number_3()
    # df = create_dataset()
    # df.to_csv("emotion_data.csv")
    # experiment_number_4()
    # experiment_number_5_librosa()
    avg = get_average_file_len()
    generate_image_files(avg)
    