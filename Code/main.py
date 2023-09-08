import torchaudio
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
audio_model = ap.set_audio_model()

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

# my_text = ("Infatti, se il costo della vita è diverso, la stessa cifra concessa come aiuto può avere un impatto "
#           "concreto molto diverso, e non vogliamo generare ulteriori distorsioni nel mercato unico.")
path = ("/home/one/.cache/huggingface/datasets/downloads/extracted"
        "/b7a3c66e6026620cda6b8044f8e8ddcec67315d2007a24718ec3e3bc533b4020/test_part_0/20180528-0900-PLENARY-20"
        "-it_20180528-20:28:03_10.wav")

wave, _ = torchaudio.load("../prove_audio/prova.wav")
audio_arr = wave.numpy()
emb = ap.get_audio_embedding(audio_arr, audio_model)

for i in range(len(emb)):
    print(f'{emb[i]},')

# sample_embedding = tp.get_text_embedding(my_text, device, txt_tokenizer, txt_model)
# qt.similarity_query(my_text, sample_embedding, text_collection, graph)


# ------------------------------------- Mixed Query: gender + similarity --------------------------------------
# feminist_text = "le donne devono denunciare gli sfruttamenti rispetto delle donne le donne le donne le donne"
# immigration_text = "fermiamo l'immigrazione salvini sei una persona ignobile" job_text = "lavoro e imprese e cose
# varie solo per allungare il testo ma vai via buffone ciao sono io come stai ao dai roma forza napoli"
# sample_embedding = tp.get_text_embedding(feminist_text, device, txt_tokenizer, txt_model)
# qt.mixed_query(graph, feminist_text, sample_embedding, text_collection, {'gender': 'male'})

# ------------------------------------- Mixed Query: gender + timestamps --------------------------------------
# feminist_text = "le donne devono denunciare gli sfruttamenti rispetto delle donne le donne le donne le donne"
# qt.get_id_and_text_by_properties_between_timestamps_neo4j(graph,
#                                                          {'gender': 'male'},
#                                                          "2013-05-20T18:11:55",
#                                                          "2015-05-20T18:11:55")
