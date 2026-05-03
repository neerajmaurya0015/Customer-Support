from sentence_transformers import SentenceTransformer
from scipy.spatial import distance

model = SentenceTransformer("all-MiniLM-L6-v2")

def validate_response(query, response, docs, threshold=0.7):
    resp_emb = model.encode(response)

    sims = []
    for d in docs:
        doc_emb = model.encode(d)
        sim = 1 - distance.cosine(resp_emb, doc_emb)
        sims.append(sim)

    if not sims:
        return False

    return max(sims) >= threshold