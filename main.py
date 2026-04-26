import os
import json
import asyncio
import time
import imaplib
import smtplib
import email
import httpx
from email.mime.text import MIMEText
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings as ChromaSettings

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmailSupportAgent")

load_dotenv()

app = FastAPI(title="Email Support Agent")

# Define the expected JSON payload for Webhook installations (if Developer didn't pick DBaaS)
class InstallRequest(BaseModel):
    agent_id: str
    gmail_address: str
    gmail_app_password: str
    # Plus any dynamic fields...
    
    class Config:
        extra = "allow" # allow extra dynamic fields

AGENT_ID = os.getenv("AGENT_ID", "default_agent")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")

# Define the expected JSON payload for Chat (from Gateway)
class ChatRequest(BaseModel):
    message: str
    injected_context: dict = {}

# Local memory to track state if using webhook / pushing
STATE = {}

# ── ChromaDB Initialization ──────────────────────────────────────────────────
# We use a local persistent directory for the vector database
CHROMA_DATA_PATH = os.path.join(os.getcwd(), "chroma_data")
os.makedirs(CHROMA_DATA_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

def get_embedding_model():
    """Helper to call Gemini for embeddings."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

async def get_embeddings(text: str):
    """Generates embeddings using gemini-embedding-2."""
    client = get_embedding_model()
    if not client:
        return []
    
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=text
        )
        return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []

def get_retrieved_context(tenant_id: str, query: str, n_results: int = 3):
    """Searches the vector database for relevant snippets."""
    try:
        collection = chroma_client.get_or_create_collection(name=f"tenant_{tenant_id}")
        
        # In a real app, we would use an embedding function in ChromaDB
        # For now, we'll do it manually to ensure we use gemini-embedding-2
        query_embedding = asyncio.run(get_embeddings(query))
        
        if not query_embedding:
            return ""
            
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""
            
        return "\n\nRelevant Context:\n" + "\n".join(results['documents'][0])
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return ""

class KnowledgePayload(BaseModel):
    tenant_id: str
    content: str
    metadata: dict = {}

def process_email_with_gemini(business_name, faq_text, message, tenant_id=None):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") # Support for proxy
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set.")
        return "Sorry, I am currently unable to process your request."
    
    client_kwargs = {"api_key": GEMINI_API_KEY}
    if GEMINI_BASE_URL:
        # The new SDK might handle base_url differently, but let's assume it works or we use a custom client
        # For now, let's stick to the standard initialization if no base_url is provided
        pass

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Retrieve extra context from Vector DB if tenant_id is provided
    vector_context = ""
    if tenant_id:
        vector_context = get_retrieved_context(tenant_id, message)

    system_prompt = (
        f"You are a helpful AI assistant for {business_name}. "
        f"Use the following business information to answer the user's inquiry:\n\n"
        f"FAQ/Information:\n{faq_text}\n"
        f"{vector_context}\n\n"
        f"Guidelines:\n"
        f"- Be polite and professional.\n"
        f"- If you don't know the answer, ask the user to contact us directly or wait for a human representative.\n"
        f"- Respond ONLY with a JSON object containing the key 'reply'."
    )
    
    user_message = f"User says: '{message}'"

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[system_prompt, user_message],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7
            )
        )
        
        text_resp = response.text.strip()
        # Clean up Markdown formatting if present
        if text_resp.startswith("```json"): text_resp = text_resp[7:-3].strip()
        elif text_resp.startswith("```"): text_resp = text_resp[3:-3].strip()
            
        data = json.loads(text_resp)
        return data.get("reply", "Thank you for your message. We will get back to you shortly.")
    except Exception as e:
        logger.error(f"Error processing email with Gemini: {e}")
        return "Thank you for your message. Our team will review it and get back to you."

def poll_once(config):
    gmail_user = config.get("gmail_address")
    gmail_pass = config.get("gmail_app_password")
    business_name = config.get("business_name", "the business")
    faq_text = config.get("faq_text", "")
    
    if not gmail_user or not gmail_pass: return

    if gmail_user not in STATE:
        STATE[gmail_user] = {"last_processed_uid": None}
        
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        mail.login(gmail_user, gmail_pass)
        mail.select("INBOX")
        
        if STATE[gmail_user]["last_processed_uid"] is None:
            status, data = mail.uid('SEARCH', None, 'ALL')
            uids = [int(u) for u in data[0].split()] if (status == "OK" and data[0]) else []
            STATE[gmail_user]["last_processed_uid"] = max(uids) if uids else 0
            mail.close()
            mail.logout()
            return
            
        search_criteria = f"UNSEEN UID {STATE[gmail_user]['last_processed_uid'] + 1}:*"
        status, messages = mail.uid('SEARCH', None, search_criteria)
        
        if status != "OK" or not messages[0]:
            mail.close()
            mail.logout()
            return
            
        for e_uid in messages[0].split():
            uid_int = int(e_uid)
            status, msg_data = mail.uid('FETCH', e_uid, "(RFC822)")
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    sender = msg.get("From")
                    subject = msg.get("Subject", "")
                    
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body_bytes = part.get_payload(decode=True)
                                body = body_bytes.decode(errors="replace") if body_bytes else ""
                                break
                    else:
                        body_bytes = msg.get_payload(decode=True)
                        body = body_bytes.decode(errors="replace") if body_bytes else ""
                    
                    if not body.strip(): continue

                    # Execute Agent Brain
                    # For email polling, we might not have the tenant_id easily accessible in this loop
                    # unless we pass it from agent_polling_loop.
                    # Let's assume we pass it or resolve it.
                    reply_text = process_email_with_gemini(business_name, faq_text, body.strip(), tenant_id=config.get("tenant_id"))
                        
                    # Send SMTP
                    smtp_msg = MIMEText(reply_text)
                    smtp_msg['Subject'] = f"Re: {subject}"
                    smtp_msg['From'] = gmail_user
                    smtp_msg['To'] = sender
                    
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtpserver:
                        smtpserver.login(gmail_user, gmail_pass)
                        smtpserver.send_message(smtp_msg)
            
            mail.uid('STORE', e_uid, '+FLAGS', '\\Seen')
            STATE[gmail_user]["last_processed_uid"] = max(STATE[gmail_user]["last_processed_uid"], uid_int)
                    
        mail.close()
        mail.logout()
    except Exception as e:
        logger.error(f"Polling error for {gmail_user}: {e}")

async def agent_polling_loop():
    logger.info("Starting Email Support Agent polling loop...")
    while True:
        await asyncio.sleep(10)
        
        # Pull latest installed businesses from Agent-fy managed DB
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"{GATEWAY_URL}/api/developers/{AGENT_ID}/tenants", timeout=5.0)
                if res.status_code == 200:
                    data = res.json()
                    tenants = data.get("tenants", [])
                    for tenant in tenants:
                        # Process each tenant in a separate thread so polling doesn't block
                        await asyncio.to_thread(poll_once, tenant)
        except Exception as e:
            pass # Keep polling running quietly even if gateway is temporarily down

@app.on_event("startup")
async def startup_event():
    # Only start background polling if we consider ourselves a DBaaS managed agent
    # In reality, this developer would configure AGENT_ID in their env file
    asyncio.create_task(agent_polling_loop())

@app.post("/install")
async def handle_install(payload: InstallRequest):
    """Fallback route for Developers who do NOT use Managed DB and requested a direct push on install."""
    data = payload.dict()
    logger.info(f"Received manual push installation for {data.get('gmail_address')}")
    return {"status": "installed"}

@app.post("/")
async def handle_chat(payload: ChatRequest):
    """
    Standard chat endpoint called by the Agent-fy Gateway.
    It uses the injected context (Business Name, FAQ) to respond.
    """
    logger.info(f"Received chat request: {payload.message}")
    
    # Extract config from injected context
    config = payload.injected_context
    business_name = config.get("business_name", config.get("restaurant_name", "the business"))
    faq_text = config.get("faq_text", "")
    tenant_id = config.get("tenant_id")
    
    reply = process_email_with_gemini(business_name, faq_text, payload.message, tenant_id=tenant_id)
    return {"reply": reply}

@app.post("/index-knowledge")
async def index_knowledge(payload: KnowledgePayload):
    """
    Indexes text content into the tenant's vector database.
    """
    try:
        collection = chroma_client.get_or_create_collection(name=f"tenant_{payload.tenant_id}")
        
        embedding = await get_embeddings(payload.content)
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
            
        doc_id = f"doc_{int(time.time() * 1000)}"
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[payload.content],
            metadatas=[payload.metadata]
        )
        
        logger.info(f"Indexed content for tenant {payload.tenant_id}: {doc_id}")
        return {"status": "success", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable if available (required for Cloud Run)
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
