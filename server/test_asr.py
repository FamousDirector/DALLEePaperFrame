# https://colab.research.google.com/github/PyThaiNLP/tutorials/blob/master/source/notebooks/thai_wav2vec2_onnx.ipynb#scrollTo=uBY5QBKd0O-j

import onnxruntime
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps

input_size = 100000
new_rate = 16000
AUDIO_MAXLEN = input_size

ort_session = onnxruntime.InferenceSession('asr.onnx')  # load onnx model

# wget https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/vocab.json

with open("vocab.json", "r", encoding="utf-8-sig") as f:
    d = eval(f.read())
res = dict((v, k) for k, v in d.items())


def normalize(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))


def remove_adjacent(item):  # code from https://stackoverflow.com/a/3460423
    nums = list(item)
    a = nums[:1]
    for item in nums[1:]:
        if item != a[-1]:
            a.append(item)
    return ''.join(a)


def asr(path):

    # read wav file
    sampling_rate, data = wavfile.read(path)

    # preprocess
    samples = round(len(data) * float(new_rate) / sampling_rate)
    new_data = sps.resample(data, samples)
    speech = np.array(new_data, dtype=np.float32)
    speech = speech.sum(axis=1) / 2  # stereo to mono
    speech = normalize(speech)[None]

    # run onnx model inference
    ort_inputs = {"modelInput": speech}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0].shape)
    prediction = np.argmax(ort_outs, axis=-1)

    # Text post-processing
    _t1 = ''.join([res[i] for i in list(prediction[0][0])])
    return ''.join([remove_adjacent(j) for j in _t1.split("<pad>")]).replace("|", " ").lower()


if __name__ == "__main__":
    print(asr("output.wav"))
