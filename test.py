import numpy
from scipy.fftpack import dct
import scipy.io.wavfile
def main():
    sample_rate, signal = scipy.io.wavfile.read('adversarial3.wav')  # File assumed to be in the same directory
    signal = signal[0:int(2 * sample_rate)]  # Keep the first 2 seconds

    pre_emphasis = 0.97
    frames_num = 320
    frame_length = int(sample_rate * 0.025)
    frame_stride = int(sample_rate * 0.01)

    signal_length = len(signal)
    signal = numpy.concatenate((signal[:1], signal[1:] - pre_emphasis * signal[:-1], numpy.zeros(frames_num, dtype=numpy.float32)), 0)

    windowed = numpy.stack([signal[i:i + frame_length] for i in range(0, signal_length - frames_num, frame_stride)], 1)

    NFFT = 512
    ffted = numpy.fft.rfft(windowed, 512)
    pow_frames = 1.0 / NFFT * numpy.square(numpy.abs(ffted))

    fbank = numpy.load("filterbanks.npy")
    filter_banks = numpy.dot(pow_frames, fbank.T) + 1e-30
    feat = numpy.log(filter_banks)
    feat = dct(feat, type=2, norm='ortho')[:,:26]
    nframes, ncoeff = feat.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (22 / 2.) * numpy.sin(numpy.pi * n / 22)
    feat = lift * feat
    engergy = numpy.sum(pow_frames,1)
    engergy = numpy.log(engergy)
    features = numpy.concatenate((engergy.reshape(-1,1),feat[:,1:]),1)
    return features


if __name__ == "__main__":
    main()
