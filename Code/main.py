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

# ------------------------------------- Query on Milvus: text similarity --------------------------------------
# racist_text = "io non sono razzista ma lo sanno tutti che gli immigrati rubano il nostro lavoro sbarcando qui"
# meat_text = ("Egregio Presidente e stimati membri dell'assemblea, la scelta di imporre l'obbligo di etichettare la "
#             "carne trasformata contenuta negli alimenti di uso comune costituisce un risultato significativo "
#             "nell'assicurare una tracciabilità superiore, prevenire frodi alimentari con conseguenze gravi per i "
#             "cittadini, e agevolare le aziende alimentari nella selezione di fornitori e prodotti di qualità "
#             "superiore.")
# egypt_text = "è drammatica la situazione in egitto"
# sample_embedding = tp.get_text_embedding(egypt_text, device, text_tokenizer, text_model)
# qt.similarity_query(text_collection, graph, egypt_text, sample_embedding, sample_type="text")


# ------------------------------------- Query on Milvus: audio similarity --------------------------------------
# audio_text = ("Infatti, se il costo della vita è diverso, la stessa cifra concessa come aiuto può avere un impatto "
#           "concreto molto diverso, e non vogliamo generare ulteriori distorsioni nel mercato unico.")

# audio_path = ("/home/one/.cache/huggingface/datasets/downloads/extracted"
#             "/b7a3c66e6026620cda6b8044f8e8ddcec67315d2007a24718ec3e3bc533b4020/test_part_0/20180528-0900-PLENARY-20"
#             "-it_20180528-20:28:03_10.wav")
#
# wave, _ = torchaudio.load("../audio_tests/20130520-0900-PLENARY-11-it_20130520-17:18:58_1.wav")
# audio_arr = wave.numpy()
# audio_arr = wave.numpy()
# embedding = ap.get_audio_embedding(audio_arr, audio_model)
#
# for i in range(len(embedding)):
#     print(f'{embedding[i]},')

# ------------------------------------- Query on Neo4J: properties  --------------------------------------
qt.properties_query(graph, {
                                'gender': 'male',
                                'timestamp_start': '2010-05-20T18:11:55',
                                'timestamp_end': '2015-05-20T18:11:55'
})

# ------------------------------------- Mixed Query: properties + similarity --------------------------------------
feminist_text = "le donne devono denunciare gli sfruttamenti rispetto delle donne le donne le donne le donne"
immigration_text = "fermiamo l'immigrazione salvini sei una persona ignobile"
job_text = "lavoro e imprese e cose varie solo per allungare il testo ma vai via buffone ciao sono io come stai ao dai roma forza napoli"
meat_text = "etichettatura della carne"
sample_embedding = tp.get_text_embedding(meat_text, device, text_tokenizer, text_model)
qt.mixed_query(text_collection, graph, meat_text, sample_embedding, {
                                                                        'gender': 'female',
                                                                        'timestamp_start': '2015-05-20T18:11:55',
                                                                        'timestamp_end': '2015-05-20T18:11:55'
                                                                    },
                                                                    sample_type="text")
