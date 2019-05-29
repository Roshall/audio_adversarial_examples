import numpy as np
import tensorflow as tf

import scipy.io.wavfile as wav
import pickle

from timeit import default_timer as timer

from attack import *

in_path = 'original_audio/'
save_path = 'original_to_adv/'
target = 'we are always happy'
iterations = 200
pickle_path = 'pickle/original_to_adv.pickle'

def main():
    audios = []
    lengths = []
    output_audio_paths = []

    dir_list = os.listdir(in_path)
    dir_list.sort()
    dir_list = dir_list[116:117]
    for file in dir_list:
        file_path = os.path.join(in_path, file)
        output_audio_paths.append(save_path + 'noise_robust_adv_' + file)
        fs, audio = wav.read(file_path)
        audios.append(list(audio))
        lengths.append(len(audio))
    config = tf.ConfigProto(log_device_placement=False,
                              allow_soft_placement=False)
    with tf.Session(config=config) as sess:
        maxlen = max(map(len,audios))
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])

        start_attack = timer()
        attack = Attack(sess, len(target), maxlen, batch_size=len(audios),
                        learning_rate=100, num_iterations=iterations, noise_robust=True)
        delta = attack.attack(audios, lengths, [[toks.index(x) for x in target]]*len(audios), noise_robust=True)

    for i in range(len(output_audio_paths)):
        wav.write(output_audio_paths[i], 16000,
                  np.array(np.clip(np.round(delta[i][:lengths[i]]),
                                   -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
    # adv_out = np.array(np.clip(np.round(delta),
    #                            -2 ** 15, 2 ** 15 - 1), dtype=np.int16)
    # np.save('adv_out.npy',adv_out)
    end_attack = timer() - start_attack
    print('attacking time: %0.3f s' % end_attack)

    # with open(pickle_path, 'wb') as f:
    #     pickle.dump([deltas, resultes1, resultes2, distortions], f, -1)
    # with tf.Session() as sess:
    #     maxlen =
    #     lengths = [maxlen]
    #     start_attack = timer()
    #     attack = Attack(sess, 'CTC', len(target), maxlen,
    #                     batch_size=1,
    #                     num_iterations=iterations)
    #     delta, res1, res2 = attack.attack(audio.reshape(1, -1), lengths,
    #                                       [[toks.index(x) for x in target]])
    #     end_attack = timer() - start_attack
    #     wav.write('test.wav', fs,
    #               np.array(np.clip(np.round(delta),
    #                                -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
    #     distortion = np.max(np.abs(delta - audio))
    # print('attacking time: %0.3fs',end_attack)

if __name__ == '__main__':
    main()