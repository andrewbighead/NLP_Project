import formatter as f
from tqdm import tqdm
import audio_processing as ap
import text_processing as tp
import numpy as np


def extract_embedding(dataset, device, model_audio, txt_model, txt_tokenizer):
    f.my_print('Starting embedding extraction...')
    audio_embeddings = []
    text_embeddings = []
    for i, item in enumerate(tqdm(dataset, desc="Extracted embeddings", total=len(dataset))):
        intervention_id = item['audio_id']
        processed_embedding_audio = ap.get_audio_embedding(item['audio']['array'], model_audio)
        audio_embeddings.append({"intervention_id": intervention_id, "audio_embedding": processed_embedding_audio})
        processed_embedding_text = tp.get_text_embedding(item['normalized_text'], device, txt_tokenizer, txt_model)
        text_embeddings.append({"intervention_id": intervention_id, "text_embedding": processed_embedding_text})
    f.my_print('Extraction completed!')
    return audio_embeddings, text_embeddings


def split_embedding_list(embeddings, n_split):
    embedding_parts = np.array_split(embeddings, n_split)
    list_of_arrays_embedding_parts = [arr.tolist() for arr in embedding_parts]
    return list_of_arrays_embedding_parts
