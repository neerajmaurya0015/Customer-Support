from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(query, docs):
    context = "\n".join(docs)

    prompt = f"""
    You are a SaaS support assistant.

    Query: {query}

    Context:
    {context}

    Answer clearly and politely.
    """

    res = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content