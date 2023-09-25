import time
import warnings
import torch
import audio_processing as ap
import dataset_process as dp
import formatter as f
import milvus_manager as mm
import neo4j_manager as n4m
import processing as p
import text_processing as tp


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    start_time = time.time()

    # ------------------------------------- Milvus -------------------------------------
    milvus_host, milvus_port = mm.get_milvus_parameter('../config/connect_milvus.json')
    mm.milvus_connect(milvus_host, milvus_port)
    audio_schema, text_schema = mm.create_schemas()
    audio_collection, text_collection = mm.create_collections(audio_schema, text_schema)

    # ------------------------------------- Neo4j -------------------------------------
    uri, user, pwd = n4m.get_neo4j_parameter('../config/connect_neo4j.json')
    n4j_conn = n4m.py2neo_connect(uri, user, pwd)

    # ------------------------------------- Dataset -------------------------------------
    dataset = dp.extract_dataset()

    # ------------------------------------- Models -------------------------------------
    model_audio = ap.set_audio_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt_tokenizer, txt_model = tp.set_text_model(device)

    # ------------------------------------- Processing -------------------------------------
    audio_embeddings, text_embeddings = p.extract_embedding(dataset, device, model_audio, txt_model, txt_tokenizer)

    # ------------------------------------- Insert in Neo4j -------------------------------------
    n4m.insert_dataset(dataset, n4j_conn)

    # ------------------------------------- Insert in Milvus -------------------------------------
    # Splitting embeddings list for a better hardware resources management during Milvus inserting
    embedding_split = 20

    list_of_arrays_audio_embedding_parts = p.split_embedding_list(audio_embeddings, embedding_split)
    list_of_arrays_text_embedding_parts = p.split_embedding_list(text_embeddings, embedding_split)

    mm.insert_dataset(audio_collection, dataset, embedding_split, list_of_arrays_audio_embedding_parts,
                      list_of_arrays_text_embedding_parts, text_collection)

    index_struct = mm.create_index_structure()
    mm.create_index_collection(audio_collection, "audio_embedding", index_struct)
    mm.create_index_collection(text_collection, "text_embedding", index_struct)

    mm.milvus_disconnect()

    # ------------------------------------- Closing -------------------------------------

    end_time = time.time() - start_time
    f.my_print(f'Total time spent: {end_time / 60} minutes.')

    f.my_print('Operations completed successfully! '
               'Wait for the processes to terminate. This may take a few more seconds.')


if __name__ == "__main__":
    main()
