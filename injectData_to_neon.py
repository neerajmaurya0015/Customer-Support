# injectData_to_neon.py
from old_llm_function import get_db_connection
conn = get_db_connection()

from sentence_transformers import SentenceTransformer
from pgvector.psycopg import Vector
import json


#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer("all-MiniLM-L6-v2")  # output is 384-dim embeddings

docs = [
    {"content": "How do I reset my password?", "metadata": {"category": "account"}},
    {"content": "How can I update my billing information?", "metadata": {"category": "billing"}},
    {"content": "Why is my app crashing on Android?", "metadata": {"category": "technical"}}
]

with conn.cursor() as cur:
    for doc in docs:
        emb = model.encode(doc["content"]).tolist()  
        cur.execute(
            "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
            (doc["content"], Vector(emb),  json.dumps(doc["metadata"]) )
        )
    conn.commit()

print("Inserted documents with embeddings")