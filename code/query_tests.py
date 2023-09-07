import formatter as f
import neo4j_manager as n4m


def mixed_query(sample_text, sample_embedding, text_collection, properties):
    # Get intervention from Neo4J that satisfy the properties and save them ids
    retrieved_interventions_satisfying_properties = query_neo4j_by_properties(properties)
    ids_filter = []
    for intervention in retrieved_interventions_satisfying_properties:
        ids_filter.append(intervention['id'])

    # From these interventions, get the most similar from Milvus
    retrieved_similar_intervention_ids = milvus_similarity_query(
        sample_text, sample_embedding, text_collection, limit=6, ids_filter=ids_filter)

    # Get the text from the most similar interventions
    for similar_intervention_id in retrieved_similar_intervention_ids:
        for satisfying_intervention in retrieved_interventions_satisfying_properties:
            if similar_intervention_id == satisfying_intervention['id']:
                print(satisfying_intervention['text'])


def similarity_query(sample_text, sample_embedding, text_collection):
    retrieved_intervention_ids = milvus_similarity_query(sample_text, sample_embedding, text_collection, limit=3)
    for retrieved_intervention in retrieved_intervention_ids:
        retrieved_text = get_text_from_neo4j(retrieved_intervention)
        print("Retrieved: " + retrieved_text)


def milvus_similarity_query(sample_text, sample_embedding, text_collection, limit=16384, ids_filter=None):
    print(f.my_print("Start searching based on vector similarity"))
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    if ids_filter is not None:
        expr = "intervention_id in ["
        ids_as_string_of_list = ", ".join([f"'{id}'" for id in ids_filter])
        expr = expr + ids_as_string_of_list + "]"
    else:
        expr = None

    result = text_collection.search(
        [sample_embedding], "text_embedding", search_params, expr=expr, limit=limit, output_fields=["intervention_id"])

    print("Sample queried: " + sample_text)
    retrieved_intervention_ids = []
    for hits in result:
        for hit in hits:
            print(f"hit: {hit}, intervention_id field: {hit.entity.get('intervention_id')}")
            retrieved_intervention_ids.append(hit.entity.get('intervention_id'))

    return retrieved_intervention_ids


def query_neo4j_by_properties(properties):
    uri, username, password = n4m.get_neo4j_parameter('../connect_neo4j.json')
    graph = n4m.py2neo_connect(uri, username, password)

    props_string = ", ".join([f"i.{key} = ${key}" for key in properties.keys()])
    query = f"MATCH(i:InterventionNode)-[:REFERS_TO]-(t:TextNode) WHERE {props_string} RETURN i.audio_id, t.raw_text"

    try:
        result = graph.run(query, **properties)
    except Exception as e:
        f.my_print(f"Errore durante il recupero dei nodi {e}")
        return None

    retrieved_interventions = []
    for intervention in result.data():
        retrieved_interventions.append({'id': intervention['i.audio_id'], 'text': intervention['t.raw_text']})

    return retrieved_interventions


def get_text_from_neo4j(intervention_id):
    uri, username, password = n4m.get_neo4j_parameter('../connect_neo4j.json')
    graph = n4m.py2neo_connect(uri, username, password)

    query = f"MATCH(t:TextNode) WHERE t.audio_id = '{intervention_id}' RETURN t.raw_text"

    try:
        result = graph.run(query)
    except Exception as e:
        f.my_print(f"Errore durante il recupero dei nodi {e}")
        return None

    return result.data()[0]['t.raw_text']




