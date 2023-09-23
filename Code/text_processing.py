from transformers import BertTokenizer, BertModel
import re
from datetime import datetime
import torch
import numpy as np
import normalizer as nrm
import formatter as f


def set_text_model(dev):
    f.my_print("Loading BERT model for text processing...")
    f.my_print(f"Selected device for embedding extraction: {dev}")

    model_name = "dbmdz/bert-base-italian-xxl-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(dev)
    f.my_print("Model loaded!")

    return tokenizer, model


def tokenize_text(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    return indexed_tokens, segments_ids


def text_to_tensor(indexed_tokens, segments_ids, dev):
    tokens_tensor = torch.tensor([indexed_tokens]).to(dev)
    segments_tensors = torch.tensor([segments_ids]).to(dev)
    return tokens_tensor, segments_tensors


def get_hidden_states(dev, model, tokens_tensor, segments_tensors):
    with torch.no_grad():
        tokens_tensor = tokens_tensor.to(dev)
        segments_tensors = segments_tensors.to(dev)
        outputs = model(tokens_tensor, segments_tensors)

        '''
            outputs[0] è l'output del token CLS e outputs[1] è l'output di tutti gli altri token nella frase,
            quindi prendiamo outputs[2] che corrisponde all'output di tutti i layers di BERT.
        '''
        return outputs[2]


def get_sentence_embedding_from_hidden_states(hidden_states):
    """
        Per ogni layer (13, 12 di BERT + 1 di input) ho un tensore 1 x n°tokens x 768
        dove 1 è il batch (una frase) e 768 sono le hidden_units (la trasformazione dell'input in quel layer)
        quindi prendo le hidden units di ogni token del batch sul penultimo layer e faccio una media:
    """
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    if torch.cuda.is_available():
        sentence_embedding = sentence_embedding.to('cpu')
    return sentence_embedding


def get_text_embedding(text, dev, tokenizer, model):
    indexed_tokens, segments_ids = tokenize_text(text, tokenizer)
    tokens_tensor, segments_tensors = text_to_tensor(indexed_tokens, segments_ids, dev)
    hidden_states = get_hidden_states(dev, model, tokens_tensor, segments_tensors)
    embedding = nrm.normalize_embedding(np.array(get_sentence_embedding_from_hidden_states(hidden_states)))
    return embedding


def create_timestamp_from_audio_id(audio_id):
    pattern = r"(\d{8}-\d{2}:\d{2}:\d{2})"
    match = re.search(pattern, audio_id)
    if match:
        extracted_date = match.group(1)
        final_timestamp = datetime.strptime(extracted_date, "%Y%m%d-%H:%M:%S")
        formatted_timestamp = final_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        return formatted_timestamp
    else:
        return None
