import torch
from speechbrain.pretrained import EncoderClassifier
import numpy as np


def set_audio_model():

    if torch.cuda.is_available():
        return EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                              run_opts={"device": "cuda"})
    else:
        return EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa")


def get_audio_embedding(audio_arr, model):
    audio_arr = torch.tensor(audio_arr)
    embedding = model.encode_batch(audio_arr).squeeze()
    if torch.cuda.is_available():
        embedding = embedding.to('cpu')
    return np.array(embedding)
