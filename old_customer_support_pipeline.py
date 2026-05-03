import os
from dotenv import load_dotenv
load_dotenv()

from old_nonLLM_function import (
    log_ticket,
    send_email_and_log,
    #escalate_ticket_to_human  # New function
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
    conversation_history: list  # New: For multi-turn context (list of dicts: {'user': query, 'bot': response})

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
    docs = search_documents(state["user_query"], top_k=3, category=category)  # Pass history if needed for context
    print("Retrieved documents:", docs)
    return {"search_docs": docs}

def responder(state: TicketState):
    # Use history for context-aware response
    context = "\n".join([f"User: {turn['user']}\nBot: {turn.get('bot', '')}" for turn in state.get("conversation_history", [])])
    print("Context for response generation:", context)
    full_query = f"{context}\nUser: {state['user_query']}"
    print("Full query for LLM:", full_query)
    resp = generate_response(full_query, state["search_docs"])
    return {"llm_response": resp}

def critic(state: TicketState):
    send_reply, details = critic_agent(
        state["user_query"], state["llm_response"], state["search_docs"]
    )
    return {"send_mail": send_reply, "details": details}

def escalate_human(state: TicketState):
    '''
    escalate_ticket_to_human(
        state["user_query"],
        state["llm_response"],
        to_email='neerajmaurya015@gmail.com',  # Customer email
        escalated=True  # Flag as escalated
    )
    '''
    escalated = True
    print("escalated to human ticket.")
    return {}

def send_reply(state: TicketState):
    escalated = False
    send_email_and_log(
        to_email="neerajmaurya015@gmail.com",  # Fix to valid email
        query=state["user_query"],
        response=state["llm_response"],
        confidence=state["details"].get("confidence", 1.0),  # From critic
        category=state["category"]
    )
    # Append to history
    history = state.get("conversation_history", [])
    print("past conversation history:", history)
    history.append({"user": state["user_query"], "bot": state["llm_response"]})
    return {"conversation_history": history}

def decision_node(state: TicketState):
    if state["send_mail"]:
        return "send_reply"
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
    decision_node,
    {"send_reply": "send_reply", "escalate": "escalate"}
)
graph.add_edge("send_reply", END)
graph.add_edge("escalate", END)

workflow = graph.compile()

# Multi-Turn Loop (Industry-aligned: Handle cycles until resolution or escalation)
max_turns = 5  # Limit to prevent infinite loops
conversation_history = []  # Persistent across turns
escalated = False
user_email = "neerajmaurya015@gmail.com"  # Customer email

query = input("Enter your support query: ")
state = {"user_query": query, "conversation_history": conversation_history}

for turn in range(max_turns):
    response = workflow.invoke(state)
    
    if "escalated" in response:  # Check if escalated (add flag in escalate_human if needed)
        escalated = True
        print("Ticket escalated to human. Awaiting resolution...")
        # Simulate human resolution (in production, wait for agent via webhook/API)
        human_response = input("Human Agent: Enter resolution response (or 'close' to end): ")
        if human_response != 'close':
            # Close ticket with human response
            send_email_and_log(
                to_email=user_email,
                query=query,
                response=human_response,
                confidence=1.0,  # Human-approved
                category=state.get("category", "unknown")
            )
        break
    else:
        # Update history and ask for follow-up
        conversation_history = response.get("conversation_history", [])
        print("conversation history line 146:", conversation_history)
        follow_up = input("Bot response sent. Enter follow-up query (or 'done' to close): ")
        if follow_up.lower() == 'done':
            break
        state["user_query"] = follow_up  # Next turn

if not escalated:
    print("Ticket closed automatically.")