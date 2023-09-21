import formatter as f
import neo4j_manager as n4m
import milvus_manager as mm


def mixed_query(collection, graph, sample_text, sample_embedding, properties, sample_type):
    # Get intervention from Neo4J that satisfy the properties and save them ids
    retrieved_interventions_satisfying_properties = n4m.get_id_and_text_by_properties_from_neo4j(graph, properties)
    ids_filter = []
    for intervention in retrieved_interventions_satisfying_properties:
        ids_filter.append(intervention['id'])

    # From these interventions, get the most similar from Milvus
    retrieved_similar_intervention_ids = mm.milvus_similarity_query(
        collection, sample_embedding, sample_type, ids_filter=ids_filter)

    # Get the text from the most similar interventions
    f.my_print("Sample queried: " + sample_text)
    i = 0
    for similar_intervention_id in retrieved_similar_intervention_ids:
        for satisfying_intervention in retrieved_interventions_satisfying_properties:
            if similar_intervention_id == satisfying_intervention['id']:
                i += 1
                f.my_print(f"{i} - {satisfying_intervention['text']}")


def similarity_query(collection, graph, sample_text, sample_embedding, sample_type):
    retrieved_intervention_ids = mm.milvus_similarity_query(collection, sample_embedding, sample_type, limit=150)

    f.my_print("Sample queried: " + sample_text)
    i = 0
    for retrieved_intervention in retrieved_intervention_ids:
        i = i + 1
        retrieved_text = n4m.get_text_by_intervention_id_from_neo4j(graph, retrieved_intervention)
        f.my_print(f"{i} {retrieved_text}")


def properties_query(graph, properties):
    f.my_print("Retrieving interventions by properties")
    retrieved_interventions = n4m.get_id_and_text_by_properties_from_neo4j(graph, properties)

    i = 0
    for intervention in retrieved_interventions:
        i += 1
        f.my_print(f"{i} {intervention['text']}")
