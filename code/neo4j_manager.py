import formatter as f
from py2neo import Graph, Node, Relationship
from tqdm import tqdm
import text_processing as tp
import json


def py2neo_connect(uri, username, password):
    try:
        f.my_print('Connecting to Neo4J...')
        graph = Graph(uri, auth=(username, password))
        f.my_print(f'Neo4J connection established to {uri}!')
        return graph

    except Exception as e:
        f.my_print(f"Error attempting connection to Milvus: {e}")
        return None


def get_neo4j_parameter(json_path):
    try:
        f.my_print("Retrieving Neo4J connection parameters...")
        with open(json_path) as json_file:
            params = json.load(json_file)
            return params['uri'], params['username'], params['password']

    except Exception as e:
        f.my_print(f"Error during Neo4J connection parameters retrieving: {e}")
        return None


def insert_sample_nodes(graph, sample):
    timestamp = tp.create_timestamp_from_audio_id(sample['audio_id'])
    sample['timestamp'] = timestamp

    intervention_node_dict = {
        "intervention_id": sample['audio_id'],
        "language": sample['language'],
        "gender": sample['gender'],
        "speaker_id": sample['speaker_id'],
        "accent": sample['accent'],
        "timestamp": sample['timestamp']
    }

    audio_node_dict = {
        "path": sample["audio"]["path"],
        "sampling_rate": sample["audio"]["sampling_rate"]
    }

    text_node_dict = {
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


def get_text_by_intervention_id_from_neo4j(graph, intervention_id):
    query = f"MATCH (t:TextNode)-[:REFERS_TO]->(i:InterventionNode) WHERE i.intervention_id = '{intervention_id}' " \
            f"RETURN t.normalized_text"
    result = graph.run(query)
    try:
        return result.data()[0]['t.normalized_text']
    except IndexError as e:
        return f'{e}'


def get_id_and_text_by_properties_from_neo4j(graph, properties, limit=20):
    props_string = " AND ".join([f"i.{key} = ${key}" for key in properties.keys() if "timestamp" not in key])
    if "timestamp_start" in properties.keys():
        props_string = props_string + f" AND datetime(i.timestamp) >= datetime($timestamp_start)"
    if "timestamp_end" in properties.keys():
        props_string = props_string + f" AND datetime(i.timestamp) <= datetime($timestamp_end)"

    query = (f"MATCH(i:InterventionNode)<-[:REFERS_TO]-(t:TextNode) "
             f"WHERE {props_string} "
             f"RETURN i.intervention_id, t.normalized_text "
             f"LIMIT {limit}")
    try:
        result = graph.run(query, **properties)
    except Exception as e:
        f.my_print(f"Errore durante il recupero dei nodi {e}")
        return None

    retrieved_interventions = []
    for intervention in result.data():
        retrieved_interventions.append({'id': intervention['i.intervention_id'],
                                        'text': intervention['t.normalized_text']})
    return retrieved_interventions


def insert_dataset(dataset, n4j_conn):
    f.my_print('Starting data insertion in Neo4J...')
    for i, item in enumerate(tqdm(dataset, desc="Data inserted in Neo4J", total=len(dataset))):
        insert_sample_nodes(n4j_conn, item)
    f.my_print('Data correctly inserted!')
