from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import formatter as f
import json


def get_milvus_parameter(json_path):
    try:
        f.my_print(f"Recupero dei parametri di connessione Milvus")
        with open(json_path) as json_file:
            params = json.load(json_file)
            return params['host'], params['port']

    except Exception as e:
        f.my_print(f"Errore durante il recupero dei parametri di connessione Milvus {e}")
        return None


def milvus_connect(host, port):
    try:
        f.my_print('Tentativo di connessione a Milvus')
        connections.connect("default", host, port)
        f.my_print(f'Connesso al database {host}:{port}!')
        return True

    except Exception as e:
        f.my_print(f"Errore durante la connessione a Milvus {e}")
        return False


def milvus_disconnect():
    f.my_print("Disconnessione da Milvus")
    connections.disconnect("default")
    f.my_print("Disconnesso dal Database Milvus")


def create_schemas():
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
    audio_collection = Collection("audio_interventions", audio_schema, consistency_level='Strong')
    text_collection = Collection("text_interventions", text_schema, consistency_level="Strong")
    return audio_collection, text_collection


def create_indexes_for_collections(audio_collection, text_collection):
    index_structure = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    audio_collection.create_index("audio_embedding", index_structure)
    text_collection.create_index("text_embedding", index_structure)


def insert_data_in_collection(data, collection):
    collection.insert(data)
    collection.flush()


def drop_collection(collection):
    utility.drop_collection(collection.name)
    f.my_print(f"collection {collection.name} eliminata.")


def get_collection(collection_name):
    return Collection(collection_name)
