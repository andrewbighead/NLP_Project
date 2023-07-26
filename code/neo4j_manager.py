import json
from neo4j import GraphDatabase
import formatter as f


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
