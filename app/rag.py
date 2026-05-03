from sentence_transformers import SentenceTransformer
from pgvector.psycopg import Vector
from app.db import get_connection

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_docs(query, top_k=3):
    conn = get_connection()
    q_emb = model.encode(query)

    with conn.cursor() as cur:
        cur.execute("""
        SELECT content FROM documents
        ORDER BY embedding <-> %s
        LIMIT %s
        """, (Vector(q_emb), top_k))

        results = [r[0] for r in cur.fetchall()]

    conn.close()
    return results