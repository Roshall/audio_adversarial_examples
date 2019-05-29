import os
# from pydub import AudioSegment
from multiprocessing import Pool
import numpy as np
import scipy.io.wavfile as wav
import colorednoise as cn

savepath = 'original_audio/'
songs = []
resource_path = ['original_audio/','original_to_adv/']
added_noise = ['original_add_white_noise/','adv_add_gaussian_white/']
# def transtype_worker(filename):
#     song = AudioSegment.from_file(filename, format='flac')
#     des = savepath + str(os.path.basename(filename).split('.')[0]) + '.wav'
#     song.export(des,format="wav")
def get_noise_energy(snr, audio):
    Nx = len(audio)
    noise_power = 10 ** (snr / 10.0)
    noise_variance = np.sum(audio ** 2) / (Nx*noise_power)
    return np.sqrt(noise_variance)

def generate_white_noise(snr, audio):
    Nx = len(audio)
    noise = np.random.randn(Nx)
    snr = 10 ** (snr / 10.0)
    audio_power = np.sum(audio ** 2) / Nx
    noise_variance = audio_power / snr
    noise = np.sqrt(noise_variance) * noise
    return noise

def add_noise_worker(arg):
    audio_path, snr, beta, out_path = arg
    fs,audio = wav.read(audio_path)
    noise = cn.powerlaw_psd_gaussian(beta, len(audio))
    noise_energy = get_noise_energy(snr, audio)
    noise = noise_energy * noise
    wav.write(out_path, fs, noise.astype(np.int16)+audio)

# def flac_to_wave():
#     for d,sd,files in os.walk('.'):
#         for f in files:
#             src = os.path.join(d,f)
#             if not src.endswith('flac'):
#                 continue
#             songs.append(src)
#     p = Pool(processes=os.cpu_count())
#     p.map(transtype_worker, songs)

def add_white(file_path,snr,out_path):
    audio_paths =[]
    dir_list = os.listdir(file_path)
    dir_list.sort()
    dir_list = dir_list[:10]
    for file in dir_list:
        audio_path = os.path.join(file_path, file)
        audio_paths.append([audio_path,snr,out_path+'noise15_'+file])
    p = Pool(processes=os.cpu_count())
    p.map(add_noise_worker, audio_paths)


def add_noise_parallel(file_path, out_path, snr, beta=0):
    color_name={-2:'violet',-1:'blue',0:'white',1:'brownian',2:'pink'}
    audio_paths =[]
    dir_list = os.listdir(file_path)
    dir_list.sort()
    dir_list = dir_list[:1]
    for file in dir_list:
        audio_path = os.path.join(file_path, file)
        audio_paths.append([audio_path, snr, beta, 
            out_path+color_name[beta]+str(snr)+'_'+file])
    p = Pool(processes=os.cpu_count())
    p.map(add_noise_worker, audio_paths)


# add_white(resource_path[1],20,added_noise[1])
# for i in range(-2,3):
#     add_noise(resource_path[1], added_noise[1], snr=20, beta=i)
# add_noise_parallel(resource_path[1], added_noise[1], snr=20)
