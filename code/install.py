import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

import audio_processing as ap
import dataset_process as dp
import formatter as f
import milvus_manager as mm
import neo4j_manager as n4m
import text_processing as tp
import query_tests as qt


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    start_time = time.time()

    # ------------------------------------- Milvus -------------------------------------
    milvus_host, milvus_port = mm.get_milvus_parameter('../connect_milvus.json')
    mm.milvus_connect(milvus_host, milvus_port)
    f.my_print("Creazione delle collezioni Milvus")
    audio_schema, text_schema = mm.create_schemas()
    audio_collection, text_collection = mm.create_collections(audio_schema, text_schema)

    # ------------------------------------- Neo4j -------------------------------------
    uri, user, pwd = n4m.get_neo4j_parameter('../connect_neo4j.json')
    n4j_conn = n4m.neo4j_connect(uri, user, pwd)

    # ------------------------------------- Dataset -------------------------------------
    f.my_print("Estrazione del Dataset VoxPopuli-it")
    dataset = dp.extract_dataset()
    dataset = dataset.shard(num_shards=510, index=0)
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
    mm.create_indexes_for_collections(audio_collection, text_collection)
    f.my_print(f'Indici per le collezioni creati correttamente')

    # ------------------------------------- Query on Milvus --------------------------------------
    # racist_text = "io non sono razzista ma lo sanno tutti che gli immigrati rubano il nostro lavoro sbarcando qui"
    # meat_text = "etichetta della carne"
    # sample_embedding = tp.get_text_embedding(meat_text, device, txt_tokenizer, txt_model)
    # qt.similarity_query(meat_text, sample_embedding, text_collection)

    # ------------------------------------- Mixed Query: gender + similarity --------------------------------------
    # feminist_text = "le donne devono denunciare gli sfruttamenti"
    # immigration_text = "fermiamo l'immigrazione"
    # job_text = "lavoro e imprese e cose varie solo per allungare il testo ma vai via buffone ciao sono io come stai ao dai roma forza napoli"
    # sample_embedding = tp.get_text_embedding(job_text, device, txt_tokenizer, txt_model)
    # qt.mixed_query(job_text, sample_embedding, text_collection, {'gender': 'male'})

    # ------------------------------------- Delete all nodes from both databases --------------------------------------
    # n4m.drop_all_nodes(n4j_conn)
    # mm.drop_collection(audio_collection)
    # mm.drop_collection(text_collection)

    # ------------------------------------- Disconnect from Databases -------------------------------------
    mm.milvus_disconnect()
    n4m.neo4j_disconnect(n4j_conn)

    end_time = time.time() - start_time
    f.my_print(f'Tempo totale: {end_time/60} minuti.')

    f.my_print(f'Operazioni completate con successo, attendere la terminazione dei processi. Questo potrebbe '
               f'richiedere qualche altro secondo.')


if __name__ == "__main__":
    main()