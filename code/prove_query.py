import neo4j_manager as n4m
import formatter as f
import numpy as np


def get_metadata_from_json(driver, body):
    # Costruisci la stringa delle proprietà per il modello di corrispondenza
    props_string = ", ".join([f"i.{key} = ${key}" for key in body.keys()])

    # Costruisci la query Cypher con le proprietà passate
    query = (f"MATCH(i:InterventionNode)-[:REFERS_TO]-(a:AudioNode) MATCH(i:InterventionNode)-[:REFERS_TO]-("
             f"t:TextNode) WHERE {props_string} RETURN i,a,t")

    try:
        result = driver.run(query, **body)
        return result
    except Exception as e:
        f.my_print(f"Errore durante il recupero dei nodi {e}")
        return None


uri, username, password = n4m.get_neo4j_parameter('../connect_neo4j.json')
graph = n4m.py2neo_connect(uri, username, password)

json_file = {"gender": 'male'}

res = get_metadata_from_json(graph, json_file)
for record in res:
    print(record)
