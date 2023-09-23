from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import formatter as f
import json
from tqdm import tqdm


def get_milvus_parameter(json_path):
    try:
        f.my_print(f"Retrieving Milvus connection parameters...")
        with open(json_path) as json_file:
            params = json.load(json_file)
            return params['host'], params['port']

    except Exception as e:
        f.my_print(f"Error during Milvus connection parameters retrieving: {e}")
        return None


def milvus_connect(host, port):
    try:
        f.my_print('Connecting to Milvus...')
        connections.connect("default", host, port)
        f.my_print(f'Milvus connection established to {host}:{port}!')
        return True

    except Exception as e:
        f.my_print(f"Error attempting connection to Milvus: {e}")
        return False


def milvus_disconnect():
    f.my_print("Disconnecting from Milvus...")
    connections.disconnect("default")
    f.my_print("Disconnected from Milvus.")


def create_schemas():
    f.my_print("Creating Milvus schemas for collections...")

    audio_intervention_fields = [
        FieldSchema(name="intervention_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=70),
        FieldSchema(name="audio_embedding", dtype=DataType.FLOAT_VECTOR, dim=256)]

    text_intervention_fields = [
        FieldSchema(name="intervention_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=70),
        FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]

    audio_schema = CollectionSchema(audio_intervention_fields,
                                    "Schema for interventions samples represented by its audio embedding")
    text_schema = CollectionSchema(text_intervention_fields,
                                   "Schema for interventions samples represented by its text embedding")
    return audio_schema, text_schema


def create_collections(audio_schema, text_schema):
    f.my_print("Creating Milvus collections...")
    audio_collection = Collection("audio_interventions", audio_schema, consistency_level='Strong')
    text_collection = Collection("text_interventions", text_schema, consistency_level="Strong")
    return audio_collection, text_collection


def create_index_structure():
    return {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 128},
    }


def create_index_collection(collection, vect_attribute, index_structure):
    f.my_print(f'Creating collections index for {collection.name} ...')
    collection.create_index(vect_attribute, index_structure)
    f.my_print('Collection index correctly created!')


def insert_data_in_collection(data, collection):
    collection.insert(data)
    collection.flush()


def drop_collection(collection):
    utility.drop_collection(collection.name)
    f.my_print(f"Collection {collection.name} deleted.")


def get_collection(collection_name):
    return Collection(collection_name)


def milvus_similarity_query(collection, sample_embedding, sample_type, limit, ids_filter=None):
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10},
    }

    if ids_filter is not None:
        expr = "intervention_id in ["
        ids_as_list_of_strings = ", ".join([f"'{intervention_id}'" for intervention_id in ids_filter])
        expr = expr + ids_as_list_of_strings + "]"
    else:
        expr = None

    result = collection.search(
        [sample_embedding],
        f"{sample_type}_embedding",
        search_params,
        expr=expr,
        limit=limit,
        output_fields=["intervention_id"]
    )

    retrieved_interventions = []
    for hits in result:
        for hit in hits:
            hit_dict = {
                "id": hit.entity.get('intervention_id'),
                "similarity": hit.distance
            }
            retrieved_interventions.append(hit_dict)
    return retrieved_interventions


def insert_dataset(audio_collection, dataset, embedding_split, list_of_arrays_audio_embedding_parts,
                   list_of_arrays_text_embedding_parts, text_collection):
    f.my_print('Starting data insertion in Milvus...')
    progress_bar_insert = tqdm(total=len(dataset), desc='Data inserted in Milvus')
    for i in range(0, embedding_split):
        insert_data_in_collection(list_of_arrays_audio_embedding_parts[i], audio_collection)
        insert_data_in_collection(list_of_arrays_text_embedding_parts[i], text_collection)
        progress_bar_insert.update(len(list_of_arrays_text_embedding_parts[i]))
    f.my_print('Data correctly inserted!')