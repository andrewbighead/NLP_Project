import json
from neo4j import GraphDatabase
import formatter as f
import text_processing as tp


def neo4j_connect(uri, username, password):
    try:
        f.my_print(f'Tentativo di connessione al database {uri}:')
        driver = GraphDatabase.driver(uri, auth=(username, password))
        f.my_print(f'Connesso al database {uri}!')
        return driver

    except Exception as e:
        f.my_print(f"Errore durante la connessione a Neo4j {e}")
        return None


def neo4j_disconnect(driver):
    f.my_print("Disconnessione da Neo4J")
    driver.close()
    f.my_print("Disconnesso dal Database Neo4J.")


def get_neo4j_parameter(json_path):
    try:
        f.my_print("Recupero dei parametri di connessione Neo4J")
        with open(json_path) as json_file:
            params = json.load(json_file)
            return params['uri'], params['username'], params['password']

    except Exception as e:
        f.my_print(f"Errore durante il recupero dei parametri di connessione Neo4j {e}")
        return None


def insert_sample_nodes(driver, sample):
    timestamp = tp.create_timestamp_from_audio_id(sample['audio_id'])
    with driver.session() as session:
        summary = session.run(
            """
                MERGE (intervention:InterventionNode {audio_id: $audio_id, language: $language, gender: $gender, 
                                        speaker_id: $speaker_id, accent: $accent, intervention_timestamp: $timestamp})
                MERGE (audio:AudioNode {audio_id: $audio_id, path: $path, sampling_rate: $sampling_rate})
                MERGE (text:TextNode {audio_id: $audio_id, raw_text: $raw_text, normalized_text: $normalized_text,
                                       is_gold_transcript: $is_gold_transcript})
                MERGE (audio)-[:REFERS_TO]->(intervention)
                MERGE (text)-[:REFERS_TO]->(intervention)
            """,
            audio_id=sample["audio_id"],
            language=sample["language"],
            gender=sample["gender"],
            speaker_id=sample["speaker_id"],
            accent=sample["accent"],
            path=sample["audio"]["path"],
            sampling_rate=sample["audio"]["sampling_rate"],
            raw_text=sample["raw_text"],
            normalized_text=sample["normalized_text"],
            is_gold_transcript=sample["is_gold_transcript"],
            timestamp=timestamp,
        ).consume()
        return session
