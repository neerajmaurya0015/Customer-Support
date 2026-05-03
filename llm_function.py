import logging

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()

try:
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("customer-support-rag")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Failed to initialize Pinecone: {e}")
    index = None
    emb_model = None

def search_documents(query, category, top_k=3):
    if not index or not emb_model:
        print("Pinecone or embedding model not initialized. Returning empty results.")
        return []
    

    query_embedding = emb_model.encode(query).tolist()
    
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"category": {"$eq": category}}
        )
        formatted_results = [
            (match["metadata"]["text"], match["metadata"], match["score"])
            for match in results["matches"]
        ]
        print(f"Searched Pinecone for {top_k} documents in category: {category}")
        print(f"Results: {formatted_results}")
        return formatted_results
    except Exception as e:
        print(f"Pinecone query failed: {e}")
        return []


# --------- LLM functions ---------


def query_classifier(query: str) -> str:
    import os

    if not os.getenv('GROQ_API_KEY'):
       raise RuntimeError("GROQ_API_KEY not set in env")

    from langchain_groq import ChatGroq
    llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=os.environ.get('GROQ_API_KEY'),
    temperature=0.7
    ) 
    prompt = f"Classify this query into [billing, technical, account, security]: {query} . Always give one word answer "
    
    cat = llm.invoke(prompt).content
    print(cat)
    return  cat.lower()



def generate_response(query, retrieved_docs):
    import os

    if not os.getenv('GROQ_API_KEY'):
       raise RuntimeError("GROQ_API_KEY not set in env")

    from langchain_groq import ChatGroq

    llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=os.environ.get('GROQ_API_KEY'),
    temperature=0.7
    ) 

    context = "\n".join([d[0] for d in retrieved_docs]) 
    prompt = f"""
    You are a helpful SaaS support assistant.
    User query: {query}

    Relevant knowledge base context:
    {context}

    Please provide a clear, friendly, and accurate response from the relevant knowledge base , please don't hallucinate.
    """

    response = llm.invoke(prompt).content

    return response



def critic_agent(query, response, retrieved_docs):
    sim_threshold=0.75
    import os
    from sentence_transformers import SentenceTransformer

    from scipy.spatial import distance
    import re
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")  # output is 384-dim embeddings
    resp_emb = emb_model.encode(response)
    sims = [1 - distance.cosine(resp_emb, emb_model.encode(doc[0])) for doc in retrieved_docs]

    if not sims:
        return False, {"reason": "no_docs", "sims": sims}

    max_sim = max(sims)

    #Semantic grounding check
    if max_sim < sim_threshold:
        return False, {"reason": "low_similarity", "sims": sims}

    #Hallucination check (checking atleast any keyword matched or not)
    for kw in ["meet", "zoom", "login", "password", "invoice", "refund", "subscription", "account", "error", "bug", "feature"]:
        if kw in response.lower() and not any(kw in d[0].lower() for d in retrieved_docs):
            return False, {"reason": f"hallucinated_{kw}", "sims": sims}

    #Tone check
    if re.search(r"(idiot|stupid|useless)", response.lower()):
        return False, {"reason": "toxic_tone", "sims": sims}

    #Confidence check LLM
    llm_confidence = critic_llm_confidence(query,response)
    if llm_confidence is not None and llm_confidence < 0.6:
        return False, {"reason": "low_llm_confidence", "sims": sims}

    #Passed all checks
    return True, {"reason": "valid", "sims": sims}


def critic_llm_confidence(query, response):
    import os

    if not os.getenv('GROQ_API_KEY'):
       raise RuntimeError("GROQ_API_KEY not set in env")

    from langchain_groq import ChatGroq

    llm = ChatGroq(
    model="meta-llama/llama-3-8b-8192",
    api_key=os.environ.get('GROQ_API_KEY'),
    temperature=0.7
    ) 

    prompt = f"""
    On a scale of 0 to 1, how confident are you that the following response accurately addresses the user's query based on provided context? 
    User query: {query}
    Response: {response}
    Provide only a decimal number between 0 and 1.
    """

    confidence_resp = llm.invoke(prompt).content
    try:
        confidence_score = float(confidence_resp.strip())
        return confidence_score
    except ValueError:
        return None