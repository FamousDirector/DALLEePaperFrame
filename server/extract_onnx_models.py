# https://colab.research.google.com/github/PyThaiNLP/tutorials/blob/master/source/notebooks/thai_wav2vec2_onnx.ipynb

import transformers
from transformers import AutoTokenizer, Wav2Vec2ForCTC
from torchaudio.models.wav2vec2.utils import import_huggingface_model


original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
imported = import_huggingface_model(
    original)  # Build Wav2Vec2Model from the corresponding model object of Hugging Face https://pytorch.org/audio/stable/models.html#wav2vec2-0-hubert
imported.eval()  # set the model to inference mode

import torch.onnx  # https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model

input_size = 100000
AUDIO_MAXLEN = input_size

dummy_input = torch.randn(1, input_size, requires_grad=True)

torch.onnx.export(imported,  # model being run
                  dummy_input,  # model input (or a tuple for multiple inputs)
                  "asr.onnx",  # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=12,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['modelInput'],  # the model's input names
                  output_names=['modelOutput'],  # the model's output names
                  dynamic_axes={'modelInput': {0: 'batch_size', 1: 'sample_length'},  # variable length axes
                                'modelOutput': {0: 'batch_size', 1: 'predicted_text'}})  # variable length axes
