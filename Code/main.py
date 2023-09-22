import torchaudio
import query_manager as qt
import numpy as np
import audio_processing as ap
import formatter as f
import milvus_manager as mm
import neo4j_manager as n4m
import torch
import text_processing as tp
import normalize as nrm

milvus_host, milvus_port = mm.get_milvus_parameter('../config/connect_milvus.json')
mm.milvus_connect(milvus_host, milvus_port)

uri, user, pwd = n4m.get_neo4j_parameter('../config/connect_neo4j.json')
graph = n4m.py2neo_connect(uri, user, pwd)

device = "cuda" if torch.cuda.is_available() else "cpu"
text_tokenizer, text_model = tp.set_text_model(device)
audio_model = ap.set_audio_model()

text_collection = mm.get_collection("text_interventions")
text_collection.load()
audio_collection = mm.get_collection("audio_interventions")
audio_collection.load()

# ------------------------------------- Query on Milvus: text similarity --------------------------------------
#racist_text = "io non sono razzista ma lo sanno tutti che gli immigrati rubano il nostro lavoro sbarcando qui"
# meat_text = ("Egregio Presidente e stimati membri dell'assemblea, la scelta di imporre l'obbligo di etichettare la "
#             "carne trasformata contenuta negli alimenti di uso comune costituisce un risultato significativo "
#             "nell'assicurare una tracciabilità superiore, prevenire frodi alimentari con conseguenze gravi per i "
#             "cittadini, e agevolare le aziende alimentari nella selezione di fornitori e prodotti di qualità "
#             "superiore.")
# egypt_text = "è drammatica la situazione in egitto"
camion_text = "Inoltre, il regolamento prevede un meccanismo per promuovere la diffusione dei camion elettrici e a" \
              " basse emissioni, con l'obiettivo di garantire che a partire dal 2025, i costruttori siano tenuti a " \
              "raggiungere una quota minima obbligatoria di tali veicoli pari al"
sample_embedding = tp.get_text_embedding(camion_text, device, text_tokenizer, text_model)
qt.similarity_query(text_collection, graph, camion_text, sample_embedding, sample_type="text")


# ------------------------------------- Query on Milvus: audio similarity --------------------------------------
# audio_text = ("Infatti, se il costo della vita è diverso, la stessa cifra concessa come aiuto può avere un impatto "
#               "concreto molto diverso, e non vogliamo generare ulteriori distorsioni nel mercato unico.")
#
# # audio_path = "../audio_tests/sample.wav" # sample del dataset
# audio_path = "../audio_tests/cut_sample.wav"
# wave, _ = torchaudio.load(audio_path)
# audio_arr = wave.numpy()
# audio_arr = audio_arr[0]
# embedding = ap.get_audio_embedding(audio_arr, audio_model)
#
# qt.similarity_query(audio_collection, graph, audio_text, embedding, sample_type="audio")

# ------------------------------------- Query on Neo4J: properties  --------------------------------------
# qt.properties_query(graph, {
#     'gender': 'male',
#     'timestamp_start': '2010-05-20T18:11:55',
#     'timestamp_end': '2015-05-20T18:11:55'
# })

# ------------------------------------- Mixed Query: properties + similarity --------------------------------------
# racist_text = "io non sono razzista ma lo sanno tutti che gli immigrati rubano il nostro lavoro sbarcando qui"
# meat_text = ("Egregio Presidente e stimati membri dell'assemblea, la scelta di imporre l'obbligo di etichettare la "
#              "carne trasformata contenuta negli alimenti di uso comune costituisce un risultato significativo "
#              "nell'assicurare una tracciabilità superiore, prevenire frodi alimentari con conseguenze gravi per i "
#              "cittadini, e agevolare le aziende alimentari nella selezione di fornitori e prodotti di qualità "
#              "superiore.")
# # egypt_text = "è drammatica la situazione in egitto"
# sample_embedding = tp.get_text_embedding(meat_text, device, text_tokenizer, text_model)
# qt.mixed_query(text_collection, graph, meat_text, sample_embedding, {
#     'gender': 'male',
#     'timestamp_start': '2010-05-20T18:11:55',
#     'timestamp_end': '2019-05-20T18:11:55'
# }, sample_type="text")
