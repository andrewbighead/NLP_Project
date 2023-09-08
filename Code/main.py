import query_tests as qt
import numpy as np
import audio_processing as ap
import formatter as f
import milvus_manager as mm
import neo4j_manager as n4m
import torch
import text_processing as tp


milvus_host, milvus_port = mm.get_milvus_parameter('../config/connect_milvus.json')
mm.milvus_connect(milvus_host, milvus_port)

uri, user, pwd = n4m.get_neo4j_parameter('../config/connect_neo4j.json')
graph = n4m.py2neo_connect(uri, user, pwd)

device = "cuda" if torch.cuda.is_available() else "cpu"
txt_tokenizer, txt_model = tp.set_text_model(device)

text_collection = mm.get_collection("text_interventions")
text_collection.load()
audio_collection = mm.get_collection("audio_interventions")
audio_collection.load()


# ------------------------------------- Query on Milvus --------------------------------------
# racist_text = "io non sono razzista ma lo sanno tutti che gli immigrati rubano il nostro lavoro sbarcando qui"
# meat_text = ("Egregio Presidente e stimati membri dell'assemblea, la scelta di imporre l'obbligo di etichettare la "
#             "carne trasformata contenuta negli alimenti di uso comune costituisce un risultato significativo "
#             "nell'assicurare una tracciabilità superiore, prevenire frodi alimentari con conseguenze gravi per i "
#             "cittadini, e agevolare le aziende alimentari nella selezione di fornitori e prodotti di qualità "
#             "superiore.")


# sample_embedding = tp.get_text_embedding(meat_text, device, txt_tokenizer, txt_model)
# qt.similarity_query(meat_text, sample_embedding, text_collection, graph)


# ------------------------------------- Mixed Query: gender + similarity --------------------------------------
feminist_text = "le donne devono denunciare gli sfruttamenti rispetto delle donne le donne le donne le donne"
# immigration_text = "fermiamo l'immigrazione salvini sei una persona ignobile" job_text = "lavoro e imprese e cose
# varie solo per allungare il testo ma vai via buffone ciao sono io come stai ao dai roma forza napoli"
sample_embedding = tp.get_text_embedding(feminist_text, device, txt_tokenizer, txt_model)
qt.mixed_query(graph, feminist_text, sample_embedding, text_collection, {'gender': 'male'})
