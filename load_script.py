import torch
from datasets import load_dataset, concatenate_datasets
from speechbrain.pretrained import EncoderClassifier
from transformers import BertTokenizer, BertModel
import numpy as np
import re
from datetime import datetime
import time
import json
from neo4j import GraphDatabase
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)


def extract_dataset():
    ds = load_dataset("facebook/voxpopuli", "it")
    train = ds['train']
    validation = ds['validation']
    test = ds['test']
    return concatenate_datasets([train, validation, test])


def set_audio_model():
    if torch.cuda.is_available():
        return EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                              run_opts={"device": "cuda"})
    else:
        return EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa")


def set_text_model(dev):
    model_name = "dbmdz/bert-base-italian-xxl-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True)
    model = model.to(dev)
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


def get_audio_embedding(audio_arr, model):
    audio_arr = torch.tensor(audio_arr)
    embedding = model.encode_batch(audio_arr).squeeze()
    if torch.cuda.is_available():
        embedding = embedding.to('cpu')
    return np.array(embedding)


def get_text_embedding(text, dev, tokenizer, model):
    indexed_tokens, segments_ids = tokenize_text(text, tokenizer)
    tokens_tensor, segments_tensors = text_to_tensor(indexed_tokens, segments_ids, dev)
    hidden_states = get_hidden_states(dev, model, tokens_tensor, segments_tensors)
    return get_sentence_embedding_from_hidden_states(hidden_states)


def create_timestamp_from_audio_id(audio_id):
    pattern = r"(\d{8}-\d{2}:\d{2}:\d{2})"
    match = re.search(pattern, audio_id)
    if match:
        extracted_date = match.group(1)
        final_timestamp = datetime.strptime(extracted_date, "%Y%m%d-%H:%M:%S")
        return final_timestamp
    else:
        return None


def neo4j_connect(uri, username, password):
    try:
        # Connessione al database Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))

        # Restituisci l'oggetto di connessione
        return driver

    except Exception as e:
        print(f"Errore durante la connessione a Neo4j: {e}")
        return None


def neo4j_disconnect(driver):
    driver.close()


def get_neo4j_parameter(json_path):
    try:
        # Carica i parametri di connessione da un file JSON
        with open(json_path) as json_file:
            params = json.load(json_file)
            return params['uri'], params['username'], params['password']

    except Exception as e:
        print(f"Errore durante il recupero dei parametri di connessione Neo4j: {e}")
        return None


def get_milvus_parameter(json_path):
    try:
        # Carica i parametri di connessione da un file JSON
        with open(json_path) as json_file:
            params = json.load(json_file)
            return params['host'], params['port']

    except Exception as e:
        print(f"Errore durante il recupero dei parametri di connessione Neo4j: {e}")
        return None


def milvus_connect(host, port):
    try:
        # Connessione al database Milvus
        connections.connect("default", host, port)

        # Restituisci l'oggetto di connessione
        return True

    except Exception as e:
        print(f"Errore durante la connessione a Milvus: {e}")
        return False


def milvus_disconnect():
    connections.disconnect("default")


h, p = get_milvus_parameter("connect_milvus.json")
connected = milvus_connect(h, p)
if connected:
    print("OK")
    milvus_disconnect()
    print("DISCONNESSO")
else:
    print("ERROR")

'''
dataset = extract_dataset()

model_audio = set_audio_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_tokenizer, txt_model = set_text_model(device)
txt_model.eval()

i = 0
time_st = time.time()
for item in dataset:
    # Query e processing della data
    print(i)
    get_audio_embedding(item['audio']['array'], model_audio)
    get_text_embedding(item['normalized_text'], device, txt_tokenizer, txt_model)
    i = i + 1
print(f'Tot: {time.time()-time_st}')

'''
