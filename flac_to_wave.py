import os
from pydub import AudioSegment
from multiprocessing import Pool
import numpy as np
import scipy.io.wavfile as wav

savepath = 'original_audio/'
songs = []
resource_path = 'original_audio/'
added_noise = 'original_add_white_noise/'
def transtype_worker(filename):
    song = AudioSegment.from_file(filename, format='flac')
    des = savepath + str(os.path.basename(filename).split('.')[0]) + '.wav'
    song.export(des,format="wav")
def add_noise_worker(arg):
    audio_path, snr, out_path = arg
    fs,audio = wav.read(audio_path)
    Nx = len(audio)
    noise = np.random.randn(Nx)
    snr = 10 **(snr/10.0)
    audio_power = np.sum(audio**2)/ Nx
    noise_variance = audio_power / snr
    noise = np.sqrt(noise_variance) * noise
    wav.write(out_path, fs, noise.astype(np.int16)+audio)

def flac_to_wave():
    for d,sd,files in os.walk('.'):
        for f in files:
            src = os.path.join(d,f)
            if not src.endswith('flac'):
                continue
            songs.append(src)
    p = Pool(processes=os.cpu_count())
    p.map(transtype_worker, songs)

def add_white(file_path,snr,out_path):
    audio_paths =[]
    dir_list = os.listdir(file_path)
    dir_list.sort()
    dir_list = dir_list[:129]
    for file in dir_list:
        audio_path = os.path.join(file_path, file)
        audio_paths.append([audio_path,snr,out_path+'noise_'+file])
    p = Pool(processes=os.cpu_count())
    p.map(add_noise_worker, audio_paths)

add_white(resource_path,4,added_noise)