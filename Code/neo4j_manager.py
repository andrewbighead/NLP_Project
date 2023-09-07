import formatter as f
from py2neo import Graph, Node, Relationship
import re
import text_processing as tp
import json


def py2neo_connect(uri, username, password):
    try:
        f.my_print(f'Tentativo di connessione al database {uri}:')
        graph = Graph(uri, auth=(username, password))
        f.my_print(f'Connesso al database {uri}!')
        return graph

    except Exception as e:
        f.my_print(f"Errore durante la connessione a Neo4j {e}")
        return None


def get_neo4j_parameter(json_path):
    try:
        f.my_print("Recupero dei parametri di connessione Neo4J")
        with open(json_path) as json_file:
            params = json.load(json_file)
            return params['uri'], params['username'], params['password']

    except Exception as e:
        f.my_print(f"Errore durante il recupero dei parametri di connessione Neo4j {e}")
        return None


def insert_sample_nodes(graph, sample):
    timestamp = tp.create_timestamp_from_audio_id(sample['audio_id'])
    sample['timestamp'] = timestamp

    intervention_node_dict = {
        "audio_id": sample['audio_id'],
        "language": sample['language'],
        "gender": sample['gender'],
        "speaker_id": sample['speaker_id'],
        "accent": sample['accent'],
        "timestamp": sample['timestamp']
    }

    audio_node_dict = {
        "audio_id": sample["audio_id"],
        "path": sample["audio"]["path"],
        "sampling_rate": sample["audio"]["sampling_rate"]
    }

    text_node_dict = {
        "audio_id": sample["audio_id"],
        "raw_text": sample["raw_text"],
        "normalized_text": sample["normalized_text"],
        "is_gold_transcript": sample["is_gold_transcript"]
    }

    intervention_node = Node("InterventionNode", **intervention_node_dict)
    audio_node = Node("AudioNode", **audio_node_dict)
    text_node = Node("TextNode", **text_node_dict)

    tx = graph.begin()
    try:
        tx.create(intervention_node)
        tx.create(audio_node)
        tx.create(text_node)
        tx.create(Relationship(text_node, "REFERS_TO", intervention_node))
        tx.create(Relationship(audio_node, "REFERS_TO", intervention_node))
        tx.commit()
        return {"Node Created"}
    except Exception as e:
        tx.rollback()
        return {str(e)}


def drop_all_nodes(driver):
    query = "MATCH (n) DETACH DELETE n"
    driver.run(query)
    return {"Message": "Database Deleted."}
