import formatter as f
import neo4j_manager as n4m


def mixed_query(graph, sample_text, sample_embedding, text_collection, properties):
    # Get intervention from Neo4J that satisfy the properties and save them ids
    retrieved_interventions_satisfying_properties = get_id_and_text_by_properties_neo4j(graph, properties)
    ids_filter = []
    for intervention in retrieved_interventions_satisfying_properties:
        ids_filter.append(intervention['id'])

    # From these interventions, get the most similar from Milvus
    retrieved_similar_intervention_ids = milvus_similarity_query(
        sample_text, sample_embedding, text_collection, ids_filter=ids_filter)

    # Get the text from the most similar interventions
    i = 0
    for similar_intervention_id in retrieved_similar_intervention_ids:
        for satisfying_intervention in retrieved_interventions_satisfying_properties:
            if similar_intervention_id == satisfying_intervention['id']:
                i += 1
                f.my_print(f"{i} - {satisfying_intervention['text']}")


def similarity_query(sample_text, sample_embedding, text_collection, graph):
    retrieved_intervention_ids = milvus_similarity_query(sample_text, sample_embedding, text_collection, limit=15)
    i = 0
    for retrieved_intervention in retrieved_intervention_ids:
        i = i + 1
        retrieved_text = get_text_by_intervention_id_neo4j(graph, retrieved_intervention)
        f.my_print(f"{i} {retrieved_text}")


def milvus_similarity_query(sample_text, sample_embedding, text_collection, limit=16384, ids_filter=None):
    f.my_print("Start searching based on vector similarity")
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    if ids_filter is not None:
        expr = "intervention_id in ["
        ids_as_list_of_strings = ", ".join([f"'{intervention_id}'" for intervention_id in ids_filter])
        expr = expr + ids_as_list_of_strings + "]"
    else:
        expr = None

    result = text_collection.search(
        [sample_embedding],
        "text_embedding",
        search_params,
        expr=expr,
        limit=limit,
        output_fields=["intervention_id"]
    )

    f.my_print("Sample queried: " + sample_text)
    retrieved_intervention_ids = []
    for hits in result:
        for hit in hits:
            hit_dict = {
                "hit": {
                    "intervention_id": hit.entity.get('intervention_id'),
                    "distance": hit.distance
                }
            }
            f.my_print(f'{hit_dict}')
            retrieved_intervention_ids.append(hit.entity.get('intervention_id'))
    return retrieved_intervention_ids


def get_text_by_intervention_id_neo4j(graph, intervention_id):
    query = f"MATCH (t:TextNode) WHERE t.audio_id = '{intervention_id}' RETURN t.raw_text"
    result = graph.run(query)
    text_r = ''
    try:
        return result.data()[0]['t.raw_text']
    except IndexError as e:
        return ''


def get_id_and_text_by_properties_neo4j(graph, properties):
    props_string = " AND ".join([f"i.{key} = ${key}" for key in properties.keys()])
    query = (f"MATCH(i:InterventionNode)-[:REFERS_TO]-(t:TextNode) "
             f"WHERE {props_string} "
             f"RETURN i.audio_id, t.raw_text")

    try:
        result = graph.run(query, **properties)
    except Exception as e:
        f.my_print(f"Errore durante il recupero dei nodi {e}")
        return None

    retrieved_interventions = []
    for intervention in result.data():
        retrieved_interventions.append({'id': intervention['i.audio_id'], 'text': intervention['t.raw_text']})
    return retrieved_interventions


def get_id_and_text_by_properties_between_timestamps_neo4j(graph, properties, start_date, end_date):

    props_string = " AND ".join([f"i.{key} = ${key}" for key in properties.keys()])

    properties["start_date"] = start_date
    properties["end_date"] = end_date

    timestamp = f" AND datetime(i.timestamp) >= datetime($start_date) AND datetime(i.timestamp) <= datetime($end_date) "

    query = (f"MATCH(i:InterventionNode)-[:REFERS_TO]-(t:TextNode) "
             f"WHERE {props_string}{timestamp}"
             f"RETURN i.audio_id, t.raw_text "
             f"ORDER BY i.timestamp")

    f.my_print(query)

    try:
        result = graph.run(query, **properties)
    except Exception as e:
        f.my_print(f"Errore durante il recupero dei nodi {e}")
        return None

    retrieved_interventions = []
    for intervention in result.data():
        retrieved_interventions.append({'id': intervention['i.audio_id'], 'text': intervention['t.raw_text']})
    f.my_print(f'{retrieved_interventions}')
    return retrieved_interventions
