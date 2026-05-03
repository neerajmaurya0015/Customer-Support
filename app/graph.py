from langgraph.graph import StateGraph, END
from typing import TypedDict
from app.rag import search_docs
from app.llm import generate_response
from app.critic import validate_response
from app.db import update_ticket

class State(TypedDict):
    query: str
    ticket_id: int
    docs: list
    response: str
    valid: bool

def retrieve(state):
    docs = search_docs(state["query"])
    return {"docs": docs}

def respond(state):
    response = generate_response(state["query"], state["docs"])
    return {"response": response}

def critic(state):
    valid = validate_response(state["query"], state["response"], state["docs"])
    return {"valid": valid}

def send_reply(state):
    update_ticket(state["ticket_id"], "pending_customer", state["response"])
    print("✅ Sent reply")
    return {}

def escalate(state):
    update_ticket(state["ticket_id"], "escalated", state["response"])
    print("🚨 Escalated to human")
    return {}

def route(state):
    return "send_reply" if state["valid"] else "escalate"

def build_graph():
    graph = StateGraph(State)

    graph.add_node("retrieve", retrieve)
    graph.add_node("respond", respond)
    graph.add_node("critic", critic)
    graph.add_node("send_reply", send_reply)
    graph.add_node("escalate", escalate)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", "critic")

    graph.add_conditional_edges("critic", route)

    graph.add_edge("send_reply", END)
    graph.add_edge("escalate", END)

    return graph.compile()

app_graph = build_graph()