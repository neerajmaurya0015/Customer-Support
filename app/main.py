from fastapi import FastAPI
from app.graph import app_graph
from app.db import create_ticket

app = FastAPI()

@app.post("/ticket")
def create(query: str):
    ticket_id = create_ticket(query)

    app_graph.invoke({
        "query": query,
        "ticket_id": ticket_id
    })

    return {"ticket_id": ticket_id}