import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from nonLLM_function import (
    log_ticket,
    send_email_and_log,
    escalate_ticket_to_human
)
from llm_function import (
    generate_response,
    search_documents,
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
    conversation_history: list
    escalated: bool

# -------- Nodes -------
def ticket_ingest(state: TicketState):
    query = state["user_query"]
    logging.info(f"Ticket ingested with query: {query}")
    return {"user_query": query}

def classifier(state: TicketState):
    q = state["user_query"]
    cat = query_classifier(q)
    logging.info(f"Query classified as: {cat}")
    return {"category": cat}

def retriever(state: TicketState):
    category = state["category"]
    docs = search_documents(state["user_query"], category=category, top_k=3)
    logging.info(f"Retrieved documents for category {category}: {len(docs)} docs")
    return {"search_docs": docs}

def responder(state: TicketState):
    context = "\n".join([f"User: {turn['user']}\nBot: {turn.get('bot', '')}" for turn in state.get("conversation_history", [])])
    full_query = f"{context}\nUser: {state['user_query']}" if context else state["user_query"]
    logging.info(f"Generating response with full query: {full_query[:100]}...")
    resp = generate_response(full_query, state["search_docs"])
    print("Generated response:", resp)
    return {"llm_response": resp}

def critic(state: TicketState):
    send_reply, details = critic_agent(
        state["user_query"], state["llm_response"], state["search_docs"]
    )
    logging.info(f"Critic decision: send_reply={send_reply}, details={details}")
    return {"send_mail": send_reply, "details": details}

def escalate_human(state: TicketState):
    escalate_ticket_to_human(
        state["user_query"],
        state["llm_response"],
        to_email="neerajmaurya0015@gmail.com",
        escalated=True
    )
    logging.info("Ticket escalated to human")
    return {"escalated": True}

def send_reply(state: TicketState):
    send_email_and_log(
        to_email="neerajmaurya0015@gmail.com",
        query=state["user_query"],
        response=state["llm_response"],
        confidence=state["details"].get("confidence", 1.0),
        category=state["category"]
    )
    history = state.get("conversation_history", [])
    history.append({"user": state["user_query"], "bot": state["llm_response"]})
    logging.info(f"Reply sent, history updated to length {len(history)}")
    return {"conversation_history": history}

def decision_node(state: TicketState):
    return "send_reply" if state["send_mail"] else "escalate"

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
    decision_node,
    {"send_reply": "send_reply", "escalate": "escalate"}
)
graph.add_edge("send_reply", END)
graph.add_edge("escalate", END)

workflow = graph.compile()

# Multi-Turn Loop
max_turns = 5
conversation_history = []
user_email = "neerajmaurya015@gmail.com"
state = {"conversation_history": conversation_history, "escalated": False}

while True:
    query = input("Enter your support query (or 'exit' to close): ")
    if query.lower() == 'exit':
        print("Support session ended.")
        break
    
    state["user_query"] = query
    try:
        response = workflow.invoke(state)
       
        state.update(response)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        break
    
    if state.get("escalated", False):
        print("Ticket escalated to human. Awaiting resolution...")
        human_response = input("Human Agent: Enter resolution response (or 'close' to end): ")
        if human_response != 'close':
            send_email_and_log(
                to_email=user_email,
                query=query,
                response=human_response,
                confidence=1.0,
                category=state.get("category", "unknown")
            )
        print("Ticket closed after human resolution.")
        break
    else:
      
        print("conversation history:", state.get("conversation_history", []))
        follow_up = input("Bot response sent. Enter follow-up query (or 'done' to close): ")
        if follow_up.lower() == 'done':
            print("Ticket closed automatically.")
            break
        state["user_query"] = follow_up