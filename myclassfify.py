## classify.py -- actually classify a sequence with DeepSpeech
##
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import tensorflow as tf

import scipy.io.wavfile as wav
import struct

import os
import sys

sys.path.append("DeepSpeech")

try:
    import pydub
except:
    print("pydub was not loaded, MP3 compression will not work")

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True


class Wrapper:
    def __init__(self, d):
        self.d = d

    def __getattr__(self, x):
        return self.d[x]


class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)

    def __getattr__(self, x):
        return self.do_define

    def do_define(self, k, v, *x):
        self.d[k] = v


tf.app.flags = HereBeDragons()
import DeepSpeech

os.path.exists = tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None

from util.text import ctc_label_dense_to_sparse
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"
path = ['original_to_adv/', 'adv_add_gaussian_white/','original_audio/','robust_adv/']

def main():
    target_path = path[3]
    dir_list = os.listdir(target_path)
    dir_list.sort()
    audios = []
    dir_list = [dir_l for dir_l in dir_list if dir_l.find('B_o') > -1]
    # dir_list = dir_list[117:118]
    files_path = []
    for file in dir_list:
        files_path.append(os.path.join(target_path, file))
    with tf.Session() as sess:
        for i in range(len(files_path)):
            if files_path[i].split(".")[-1] == 'wav':
                _, audio = wav.read(files_path[i])
                # audios.append(list(audio))
            elif files_path[i].split(".")[-1] == 'mp3':
                raw = pydub.AudioSegment.from_mp3(files_path[i])
                audio = np.array([struct.unpack("<h", raw.raw_data[i:i+2])[0] for i in range(0,len(raw.raw_data),2)])
            else:
                raise Exception("Unknown file format")
            N = len(audio)
            # audio = audio + np.random.randn(N)* 50
            new_input = tf.placeholder(tf.float32, [1, N])
            lengths = tf.placeholder(tf.int32, [1])

            with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                logits = get_logits(new_input, lengths)

            if i == 0:
                saver = tf.train.Saver()
                saver.restore(sess, "models/session_dump")

            decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

            length = (len(audio) - 1) // 320
            r = sess.run(decoded, {new_input: [audio],
                                   lengths: [length]})
            if len(files_path[i]) > 2:
                print(files_path[i])
            print("".join([toks[x] for x in r[0].values]))
    # re1=np.load('adv_raw.npy')
    # re2=np.load('adv_out.npy')
    # maxlen = max(map(len, audios))
    # re3=np.array([x+[0]*(maxlen-len(x)) for x in audios])
def test():
    target_path = path[0]
    dir_list = os.listdir(target_path)
    dir_list.sort()
    audios = []
    dir_list = [dir_l for dir_l in dir_list if dir_l.find('huge') > -1]
    # dir_list = dir_list[117:118]
    files_path = []
    for file in dir_list:
        files_path.append(os.path.join(target_path, file))
    with tf.Session() as sess:
        if files_path[0].split(".")[-1] == 'wav':
            _, audio = wav.read(files_path[0])
            # audios.append(list(audio))
        else:
            raise Exception("Unknown file format")
        audios = []
        grap =321
        # grap = 10
        for i in range(grap):
            # padding = np.zeros(grap, dtype=np.int16)
            padding = np.random.randn(grap) * 100
            # padding = padding.astype(np.int16)
            audios.append(np.concatenate((padding[:i], audio, padding[i:])))
        N = len(audio) + grap
        new_input = tf.placeholder(tf.float32, [grap, N])
        lengths = tf.placeholder(tf.int32, [grap])

        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            logits = get_logits(new_input, lengths)

        saver = tf.train.Saver()
        saver.restore(sess, "models/session_dump")

        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)

        # length = (len(audio) - 1) // 320
        # length = [(N-grap-1+i)// 320 for i in range(grap)]
        length = [N//320] * grap
        r = sess.run(decoded, {new_input: audios,
                               lengths: length})
        res = np.zeros(r[0].dense_shape) + len(toks) - 1

        for ii in range(len(r[0].values)):
            x, y = r[0].indices[ii]
            res[x, y] = r[0].values[ii]
        res = ["".join(toks[int(x)] for x in y).replace("-", "") for y in res]

        print("\n".join(res))


main()
