from timeit import default_timer as timer
import sys
import os
import scipy.io.wavfile as wav
from deepspeech.model import Model
import pickle
# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9
pickle_path = 'pickle/classify200_original_and_adv.pickle'
models_path = 'models/'
audio_pathss = ['original_audio/','adv_add_gaussian_white/','original_to_adv/','original_add_white_noise/']
model_path = models_path + 'output_graph.pb'
alphabet_path = models_path + 'alphabet.txt'
lm_path = models_path + 'lm.binary'
trie_path = models_path + 'trie'


def main():
    print('loading model from file %s' % (model_path),file=sys.stderr)
    model_load_start = timer()
    ds = Model(model_path, N_FEATURES, N_CONTEXT, alphabet_path, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    print('Loaded model in %0.3fs.' % (model_load_end))
    print('loading language model from files %s %s' % (lm_path, trie_path))
    ds.enableDecoderWithLM(alphabet_path, lm_path, trie_path, LM_WEIGHT,
                           WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
    audio_paths = audio_pathss[::2][:2]
    results = [[] for i in range(len(audio_paths))]
    lengths = [[] for i in range(len(audio_paths))]
    i = 0
    for audio_path in audio_paths:
        dir_list = os.listdir(audio_path)
        dir_list.sort()
        dir_list = dir_list[:200]
        for file in dir_list:
            file_path = os.path.join(audio_path, file)
            fs, audio = wav.read(file_path)
            audio_length = len(audio) * (1 / 16000)

            # inference_start = timer()
            # print("Running inference...")
            results[i].append(ds.stt(audio, fs))
            lengths[i].append(audio_length)
            # inference_end = timer() - inference_start
            # print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))
        i += 1
    with open(pickle_path, 'wb') as f:
        pickle.dump([results,lengths], f, -1)

if __name__ == '__main__':
    main()