import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, From
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import time
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


try:
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index("customer-support-rag")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logging.error(f"Failed to initialize Pinecone or SentenceTransformer: {e}")
    index = None
    emb_model = None

def log_ticket(user_query, llm_response, confidence, param4, category):
    """
    Log ticket data to Pinecone for future RAG retrieval.
    """
    if not index or not emb_model:
        logging.warning("Pinecone or embedding model not initialized. Skipping ticket logging.")
        return
    
    logging.info("Logging ticket to Pinecone...")
    
    ticket_content = f"Query: {user_query}\nResponse: {llm_response}"
    try:
        embedding = emb_model.encode(ticket_content).tolist()
        ticket_id = f"ticket_{hash(user_query)}_{int(time.time())}"
        
        metadata = {
            "text": ticket_content,
            "category": category,
            "timestamp": int(time.time()),
            "status": "closed" if category != "escalated" else "escalated"
        }
        
        index.upsert(vectors=[{
            "id": ticket_id,
            "values": embedding,
            "metadata": metadata
        }])
        logging.info(f"Stored ticket {ticket_id} in Pinecone with category: {category}")
    except Exception as e:
        logging.error(f"Failed to upsert ticket to Pinecone: {e}")

def send_email_and_log(to_email, query, response, confidence, category):

    logging.info("Sending email via SendGrid and logging to Pinecone...")
    
    subject = f"Support Response: {category.capitalize()} Inquiry"
    html_content = (
        f"<p>Dear Customer,</p>"
        f"<p>Thank you for reaching out with your query: <strong>{query}</strong></p>"
        f"<p>{response.replace('\n', '<br>')}</p>"
        f"<p>If you need further assistance, please reply or contact us at "
        f"<a href='mailto:support@my_email.com'>support@my_email.com</a>.</p>"
        f"<p>Best regards,<br>Your Support Team</p>"
    )
    
    message = Mail(
        from_email=From('neerajmaurya015@gmail.com'),
        to_emails=to_email,
        subject=subject,
        html_content=html_content
    )
    message.reply_to = 'support@my_email.com'
    
    try:
        sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        sg_response = sg.send(message)
        logging.info(f"Email sent to {to_email} with status: {sg_response.status_code}")
    except Exception as e:
        logging.error(f"Email sending failed: {e}")
        if hasattr(e, 'body'):
            logging.error(f"Error details: {e.body}")
    
    log_ticket(query, response, confidence, True, category=category)
    logging.info(f"Email sent to {to_email} and ticket logged.")

def escalate_ticket_to_human(user_query, llm_response, to_email, escalated):
  
    agent_email = 'neerajmaurya1017@gmail.com'
    category = "escalated"
    
    logging.info("Escalating ticket to human...")
    
    # Notify customer
    customer_subject = "Your Support Ticket Has Been Escalated"
    customer_body = (
        f"<p>Dear Customer,</p>"
        f"<p>Your query: <strong>{user_query}</strong></p>"
        f"<p>Initial response: {llm_response.replace('\n', '<br>')}</p>"
        f"<p>We've escalated this to a specialist. Expect a response within 24 hours.</p>"
        f"<p>Best regards,<br>Support Team</p>"
    )
    message_to_customer = Mail(
        from_email=From('neerajmaurya015@gmail.com', 'Support Team'),
        to_emails=to_email,
        subject=customer_subject,
        html_content=customer_body
    )
    message_to_customer.reply_to = 'support@my_email.com'
    
    
    # Notify agent
    agent_subject = "New Escalated Ticket"
    agent_body = (
        f"<p>Escalated Ticket:</p>"
        f"<p>Customer Email: {to_email}</p>"
        f"<p>Query: <strong>{user_query}</strong></p>"
        f"<p>LLM Response: {llm_response.replace('\n', '<br>')}</p>"
        f"<p>Category: {category}</p>"
        f"<p>Please resolve and reply to customer, then log resolution.</p>"
    )
    
    message_to_agent = Mail(
        from_email=From('neerajmaurya015@gmail.com', 'Support Team'),
        to_emails=agent_email,
        subject=agent_subject,
        html_content=agent_body
    )
    message_to_agent.reply_to = to_email
    
    try:
        sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
        sg.send(message_to_customer)
        sg.send(message_to_agent)
        logging.info(f"Escalation emails sent to {to_email} and {agent_email}.")
    except Exception as e:
        logging.error(f"Escalation email failed: {e}")
        if hasattr(e, 'body'):
            logging.error(f"Error details: {e.body}")
    
    log_ticket(user_query, llm_response, 0.0, False, category="escalated")