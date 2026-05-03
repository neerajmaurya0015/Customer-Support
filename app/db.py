import psycopg
import os
from pgvector.psycopg import register_vector

def get_connection():
    conn = psycopg.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)
    return conn

def create_ticket(query):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO tickets (query, status) VALUES (%s, 'open') RETURNING id",
            (query,)
        )
        ticket_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return ticket_id

def update_ticket(ticket_id, status, response=None):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE tickets SET status=%s, response=%s WHERE id=%s",
            (status, response, ticket_id)
        )
    conn.commit()
    conn.close()