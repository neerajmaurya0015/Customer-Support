from sentence_transformers import SentenceTransformer
from pgvector.psycopg import Vector
from app.db import get_connection

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "To reset password click forgot password",
    "Billing issues can be solved in billing dashboard",
    "App crash fix: reinstall app"
]

conn = get_connection()

with conn.cursor() as cur:
    for d in docs:
        emb = model.encode(d).tolist()
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (d, Vector(emb))
        )

conn.commit()
conn.close()