import numpy as np


def normalize_embedding(emb):
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    else:
        normalized_arr = emb / norm
        return normalized_arr
