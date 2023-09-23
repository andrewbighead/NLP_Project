import formatter as f
import neo4j_manager as n4m
import milvus_manager as mm


def mixed_query(collection, graph, sample_text, sample_embedding, properties, sample_type):
    f.my_print("Retrieving interventions by similarity and properties...")
    f.my_print(f"Query input: {sample_text}")
    f.my_print(f"Requested intervention properties: {properties}")

    # Get intervention from Neo4J that satisfy the properties and save them ids
    retrieved_interventions_satisfying_properties = n4m.get_id_and_text_by_properties_from_neo4j(graph, properties)
    ids_filter = []
    for intervention in retrieved_interventions_satisfying_properties:
        ids_filter.append(intervention['id'])

    # From these interventions, get the most similar from Milvus
    retrieved_similar_interventions = mm.milvus_similarity_query(
        collection, sample_embedding, sample_type, ids_filter=ids_filter)

    # Get the text from the most similar interventions
    i = 0
    for similar_intervention in retrieved_similar_interventions:
        for satisfying_intervention in retrieved_interventions_satisfying_properties:
            if similar_intervention['id'] == satisfying_intervention['id']:
                i += 1
                f.my_print(f"{i} - {satisfying_intervention['text']}\n"
                           f"Similarity: {similar_intervention['similarity']}")


def similarity_query(collection, graph, sample_text, sample_embedding, sample_type, limit=20):
    f.my_print("Retrieving interventions by similarity...")
    f.my_print(f"Query input: {sample_text}")

    retrieved_interventions = mm.milvus_similarity_query(collection, sample_embedding, sample_type, limit)

    i = 0
    for retrieved_intervention in retrieved_interventions:
        i += 1
        retrieved_text = n4m.get_text_by_intervention_id_from_neo4j(graph, retrieved_intervention['id'])
        f.my_print(f"{i} - {retrieved_text}\n"
                   f"Similarity: {retrieved_intervention['similarity']}")


def properties_query(graph, properties):
    f.my_print("Retrieving interventions by properties...")
    f.my_print(f"Requested intervention properties: {properties}")

    retrieved_interventions = n4m.get_id_and_text_by_properties_from_neo4j(graph, properties)

    i = 0
    for intervention in retrieved_interventions:
        i += 1
        f.my_print(f"{i} - {intervention['text']}")
