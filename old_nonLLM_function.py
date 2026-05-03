

import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

load_dotenv()



def send_email_and_log(to_email , query, response, confidence, category):

    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    
    print("Sending email via SendGrid and logging to Pinecone...")

    subject = f"Support Response: {category.capitalize()} Inquiry"

    body = (
        f"Dear Customer,\n\n"
        f"Thank you for reaching out with your query: \"{query}\"\n\n"
        f"{response}\n\n"
        f"If you need further assistance, please reply to this email or contact us at support@my_email.com.\n\n"
        f"Best regards,\nYour Support Team"
    )

    message = Mail(
        from_email='neerajmaurya015@gmail.com',  # Use env var or fallback
        to_emails=to_email,
        subject=subject,
        plain_text_content=body
    )

    try:
        sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        sg_response = sg.send(message)
        print(f"Email sent to {to_email} with status: {sg_response.status_code}")
    except Exception as e:
        print(f"Email sending failed: {e}")

    # Log ticket to Pinecone (existing code)
    log_ticket(query, response, confidence, True, category=category)

    print(f"Email sent to {to_email} and ticket logged.")

'''
def send_email_and_log(to_email, subject, body, query, response, confidence, category):
    """
    Send email and log closed ticket to Pinecone.
    category: Category of the ticket (passed from pipeline).
    """
    print("Sending email and logging to Pinecone...")
    
    # Simulate email sending (implement actual logic here)
    # Example: send_email(to_email=to_email, subject=subject, body=body)
    
    # Log ticket to Pinecone
    log_ticket(query, response, confidence, True, category=category)
    
    print(f"Email sent to {to_email} and ticket logged.")
'''


def log_ticket(user_query, llm_response, category):

    # Log closed ticket data to Pinecone for future RAG retrieval.
    import time
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
    import os
    from dotenv import load_dotenv
    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("customer-support-rag")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Logging ticket to Pinecone...")

    ticket_content = f"Query: {user_query}\nResponse: {llm_response}"
    embedding = emb_model.encode(ticket_content).tolist()
    ticket_id = f"ticket_{hash(user_query)}_{int(time.time())}"
    
    metadata = {
        "text": ticket_content,  # Store full content for RAG
        "category": category,    # Use ticket category or default to 'closed_ticket'
        "timestamp": int(time.time()),
        "status": "closed"       # Mark as closed for potential filtering
    }
    
    # Upsert to Pinecone
    index.upsert(vectors=[{
        "id": ticket_id,
        "values": embedding,
        "metadata": metadata
    }])
    
    print(f"Stored ticket {ticket_id} in Pinecone with category: {category}")

