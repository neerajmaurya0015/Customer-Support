import os
from dotenv import load_dotenv
load_dotenv()



from old_nonLLM_function import (
    log_ticket,
    send_email_and_log
)   
from old_llm_function import (
    generate_response,
    search_documents,
    search_documents_old,
    critic_agent,
    query_classifier
)

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class TicketState(TypedDict):
    user_query: str
    category: str
    search_docs: list
    llm_response: str
    send_mail: bool
    details: dict

# -------- Nodes -------
def ticket_ingest(state: TicketState):
    query = state["user_query"]
    return {"user_query": query}


def classifier(state: TicketState):
    q = state["user_query"]
    cat = query_classifier(q)
    return {"category": cat}


def retriever(state: TicketState):
    category = state["category"]
    if category == "billing":
        docs = search_documents(state["user_query"], top_k=3, category="billing")
    elif category == "technical":
        docs = search_documents(state["user_query"], top_k=3, category="technical")
    elif category == "account":
        docs = search_documents(state["user_query"], top_k=3, category="account")
    else:
        docs = search_documents(state["user_query"], top_k=3, category="security")
    return {"search_docs": docs}


def responder(state: TicketState):
    resp = generate_response(state["user_query"], state["search_docs"])
    return {"llm_response": resp}


def critic(state: TicketState):
    send_reply, details = critic_agent(
        state["user_query"], state["llm_response"], state["search_docs"]
        )
    return {"send_mail": send_reply, "details": details}



def escalate_human(state: TicketState):
    escalate_ticket_to_human(state["user_query"], state["llm_response"], to_email:"neerajmaurya015.com", False)
    #print("Escalated to human support")
    return {}


def send_reply(state: TicketState):
    send_email_and_log(
        to_email="neerajmaurya015.com",
        subject="Support Reply",
        body=state["llm_response"],
        query=state["user_query"],
        response=state["llm_response"],
        confidence=True,
        category=state["category"]
    )
    return {}


def decision_node(state: TicketState):
    if state["send_mail"]:
        return "send_reply1"
    else:
        return "escalate"

# ----- Build Graph -------
graph = StateGraph(TicketState)

graph.add_node("ticket_ingest", ticket_ingest)
graph.add_node("classifier", classifier)
graph.add_node("retriever", retriever)
graph.add_node("responder", responder)
graph.add_node("critic", critic)
graph.add_node("escalate", escalate_human)
graph.add_node("send_reply", send_reply)

graph.set_entry_point("ticket_ingest")

graph.add_edge("ticket_ingest", "classifier")
graph.add_edge("classifier", "retriever")
graph.add_edge("retriever", "responder")
graph.add_edge("responder", "critic")


graph.add_conditional_edges(
    "critic",
    decision_node,{ "send_reply1": "send_reply", "escalate": "escalate" }
    )

graph.add_edge("send_reply", END)
graph.add_edge("escalate", END)

workflow = graph.compile()

query = input("Enter your support query: ")
response = workflow.invoke({"user_query":query})
# print(response)

