import torch
from speechbrain.pretrained import EncoderClassifier
import numpy as np
import normalizer as nrm
import formatter as f


def set_audio_model():
    f.my_print("Loading SpeechBrain model for audio processing...")
    if torch.cuda.is_available():
        encoder = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                                 run_opts={"device": "cuda"})
        dev = "cuda"
    else:
        encoder = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa")
        dev = "cpu"
    f.my_print(f"Selected device for audio embedding extraction: {dev}")
    f.my_print("Model loaded!")

    return encoder


def get_audio_embedding(audio_arr, model):
    audio_arr = torch.tensor(audio_arr)
    embedding = model.encode_batch(audio_arr).squeeze()
    if torch.cuda.is_available():
        embedding = embedding.to('cpu')
    embedding = nrm.normalize_embedding(np.array(embedding))
    return embedding
