import torchaudio
import query_manager as qm
import audio_processing as ap
import milvus_manager as mm
import neo4j_manager as n4m
import text_processing as tp
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
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
# meat_text = ("Egregio Presidente e stimati membri dell'assemblea, la scelta di imporre l'obbligo di etichettare la "
#              "carne trasformata contenuta negli alimenti di uso comune costituisce un risultato significativo "
#              "nell'assicurare una tracciabilità superiore, prevenire frodi alimentari con conseguenze gravi per i "
#              "cittadini, e agevolare le aziende alimentari nella selezione di fornitori e prodotti di qualità "
#              "superiore.")
# egypt_text = "è drammatica la situazione in egitto"
# camion_text = "Inoltre, il regolamento prevede un meccanismo per promuovere la diffusione dei camion elettrici e a" \
#               " basse emissioni, con l'obiettivo di garantire che a partire dal 2025, i costruttori siano tenuti a " \
#               "raggiungere una quota minima obbligatoria di tali veicoli pari al"
# input_embedding = tp.get_text_embedding(egypt_text, device, text_tokenizer, text_model)
# qm.similarity_query(text_collection, graph, input_embedding, egypt_text, input_type="text", limit=5)


# ------------------------------------- Query on Milvus: audio similarity --------------------------------------
# audio_text = ("Infatti, se il costo della vita è diverso, la stessa cifra concessa come aiuto può avere un impatto "
#               "concreto molto diverso, e non vogliamo generare ulteriori distorsioni nel mercato unico.")
#
# audio_path = "../audio_tests/sample.wav"
# audio_path = "../audio_tests/cut_sample.wav"
# audio_path = "../audio_tests/sample_cut_last_second.wav"
# audio_path = "../audio_tests/rec_mario.wav"
# audio_path = "../audio_tests/voce_andrea.wav"
# wave, _ = torchaudio.load(audio_path)
# audio_arr = wave.numpy()
# audio_arr = audio_arr[0]
# input_embedding = ap.get_audio_embedding(audio_arr, audio_model)
# qm.similarity_query(audio_collection, graph, input_embedding, audio_path, input_type="audio", limit=5)


# ------------------------------------- Query on Neo4J: properties  --------------------------------------
# qm.properties_query(graph, {
#      'gender': 'male',
#      'timestamp_start': '2010-05-20T18:11:55',
#      'timestamp_end': '2015-05-20T18:11:55'
# }, limit=5)


# ------------------------------------- Mixed Query: properties + text similarity --------------------------------------
# meat_text = ("Egregio Presidente e stimati membri dell'assemblea, la scelta di imporre l'obbligo di etichettare la "
#              "carne trasformata contenuta negli alimenti di uso comune costituisce un risultato significativo "
#              "nell'assicurare una tracciabilità superiore, prevenire frodi alimentari con conseguenze gravi per i "
#              "cittadini, e agevolare le aziende alimentari nella selezione di fornitori e prodotti di qualità "
#              "superiore.")
# egypt_text = "è drammatica la situazione in egitto"
# input_embedding = tp.get_text_embedding(egypt_text, device, text_tokenizer, text_model)
# qm.mixed_query(text_collection, graph, input_embedding, {
#     'gender': 'male',
#     'timestamp_start': '2010-05-20T18:11:55',
#     'timestamp_end': '2015-05-20T18:11:55'
# }, egypt_text, input_type="text", limit=5)


# ------------------------------------- Mixed Query: properties + audio similarity ------------------------------------
# audio_text = ("Infatti, se il costo della vita è diverso, la stessa cifra concessa come aiuto può avere un impatto "
#               "concreto molto diverso, e non vogliamo generare ulteriori distorsioni nel mercato unico.")
#
# audio_path = "../audio_tests/sample.wav"
# audio_path = "../audio_tests/cut_sample.wav"
# audio_path = "../audio_tests/sample_cut_last_second.wav"
# audio_path = "../audio_tests/rec_mario.wav"
# audio_path = "../audio_tests/voce_andrea.wav"
# wave, _ = torchaudio.load(audio_path)
# audio_arr = wave.numpy()
# audio_arr = audio_arr[0]
# input_embedding = ap.get_audio_embedding(audio_arr, audio_model)
#
# qm.mixed_query(audio_collection, graph, input_embedding, {
#     'gender': 'male',
#     'timestamp_start': '2010-05-20T18:11:55',
#     'timestamp_end': '2019-05-20T18:11:55'
# }, audio_path, input_type="audio", limit=5)

# ------------------------------------- Exiting application --------------------------------------
mm.milvus_disconnect()
