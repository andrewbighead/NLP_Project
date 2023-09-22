import time
import warnings
import torch
from tqdm import tqdm
import numpy as np

import audio_processing as ap
import dataset_process as dp
import formatter as f
import milvus_manager as mm
import neo4j_manager as n4m
import text_processing as tp


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    start_time = time.time()

    # ------------------------------------- Milvus -------------------------------------
    milvus_host, milvus_port = mm.get_milvus_parameter('../config/connect_milvus.json')
    mm.milvus_connect(milvus_host, milvus_port)
    f.my_print("Creazione delle collezioni Milvus")
    audio_schema, text_schema = mm.create_schemas()
    audio_collection, text_collection = mm.create_collections(audio_schema, text_schema)

    # ------------------------------------- Neo4j -------------------------------------
    uri, user, pwd = n4m.get_neo4j_parameter('../config/connect_neo4j.json')
    n4j_conn = n4m.py2neo_connect(uri, user, pwd)

    # ------------------------------------- Dataset -------------------------------------
    f.my_print("Estrazione del Dataset VoxPopuli-it")
    dataset = dp.extract_dataset()
    f.my_print("Dataset Estratto correttamente")

    # ------------------------------------- Modelli -------------------------------------
    f.my_print("Caricamento del modello SpeechBrain per il processing audio")
    model_audio = ap.set_audio_model()
    f.my_print("Modello Caricato")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    f.my_print(f"Device selezionato per l'estrazione degli embeddings: {device}")
    f.my_print("Caricamento del modello BERT per il processing testo")
    txt_tokenizer, txt_model = tp.set_text_model(device)
    f.my_print("Modello Caricato")

    # ------------------------------------- Processing -------------------------------------
    f.my_print(f'Inizio estrazione dati')
    audio_embeddings = []
    text_embeddings = []
    for i, item in enumerate(tqdm(dataset, desc="Embedding Estratti", total=len(dataset))):
        intervention_id = item['audio_id']
        processed_embedding_audio = ap.get_audio_embedding(item['audio']['array'], model_audio)
        audio_embeddings.append({"intervention_id": intervention_id, "audio_embedding": processed_embedding_audio})
        processed_embedding_text = tp.get_text_embedding(item['normalized_text'], device, txt_tokenizer, txt_model)
        text_embeddings.append({"intervention_id": intervention_id, "text_embedding": processed_embedding_text})
    f.my_print(f'Estrazione Completata')

    # ------------------------------------- Insert in Neo4j -------------------------------------
    for i, item in enumerate(tqdm(dataset, desc="Dati inseriti in Neo4J:", total=len(dataset))):
        n4m.insert_sample_nodes(n4j_conn, item)
    f.my_print('Dati inseriti correttamente')

    # Split delle liste di embedding per gestire meglio le risorse HW durante l'inserimento in milvus
    embedding_split = 25
    audio_embedding_parts = np.array_split(audio_embeddings, embedding_split)
    text_embedding_parts = np.array_split(text_embeddings, embedding_split)
    list_of_arrays_audio_embedding_parts = [arr.tolist() for arr in audio_embedding_parts]
    list_of_arrays_text_embedding_parts = [arr.tolist() for arr in text_embedding_parts]

    # # ------------------------------------- Insert in Milvus -------------------------------------

    f.my_print(f'Inizio inserimento dati in Milvus')
    progress_bar_insert = tqdm(total=len(dataset), desc='Dati inseriti in Milvus:')
    for i in range(0, embedding_split):
        mm.insert_data_in_collection(list_of_arrays_audio_embedding_parts[i], audio_collection)
        mm.insert_data_in_collection(list_of_arrays_text_embedding_parts[i], text_collection)
        progress_bar_insert.update(len(list_of_arrays_text_embedding_parts[i]))

    f.my_print(f'Dati inseriti correttamente')
    f.my_print(f'Creazione Indici per le collezioni')
    index_struct = mm.create_index_structure()
    mm.create_index_collection(audio_collection, "audio_embedding", index_struct)
    mm.create_index_collection(text_collection, "text_embedding", index_struct)
    f.my_print(f'Indici per le collezioni creati correttamente')
    mm.milvus_disconnect()

    end_time = time.time() - start_time
    f.my_print(f'Tempo totale: {end_time / 60} minuti.')

    f.my_print(f'Operazioni completate con successo, attendere la terminazione dei processi. Questo potrebbe '
               f'richiedere qualche altro secondo.')


if __name__ == "__main__":
    main()
