"""
Email Support Agent — Production RAG-powered email responder.

This agent:
  1. Monitors a Gmail inbox for new emails via IMAP
  2. Retrieves relevant context from a ChromaDB vector store (RAG)
  3. Generates professional replies via Gemini 2.5 Flash
  4. Sends replies via SMTP

This agent is stateless; the Gateway is responsible for triggering '/sync' calls.
"""

import os
import io
import yaml
import asyncio
from firebase_admin import credentials, firestore, initialize_app
from cryptography.fernet import Fernet
import time
import imaplib
import smtplib
import email
import hashlib
import json
import httpx
from email.mime.text import MIMEText
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
import chromadb

import pypdf
import docx        # python-docx
import openpyxl

import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("EmailSupportAgent")

load_dotenv()

app = FastAPI(
    title="Email Support Agent",
    description="RAG-powered email support agent with multi-format knowledge ingestion.",
    version="3.0.0",
)

# ── Environment ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── ChromaDB Initialization ─────────────────────────────────────────────────
CHROMA_DATA_PATH = os.path.join(os.getcwd(), "knowledge_data") # Persistent storage
os.makedirs(CHROMA_DATA_PATH, exist_ok=True)

# ── Encryption & Database ───────────────────────────────────────────────────
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    # Fallback/Development key — in production, this MUST be in .env
    ENCRYPTION_KEY = b'v-9R6R_O9v6R8_U9v6R8_U9v6R8_U9v6R8_U9v6R8_U='

cipher_suite = Fernet(ENCRYPTION_KEY)

# Initialize Firebase
try:
    # Use default credentials (Service Account) if available, else look for key file
    initialize_app()
except Exception:
    pass # Already initialized or handled by environment

db = firestore.client()

def decrypt_value(encrypted_value: str) -> str:
    try:
        return cipher_suite.decrypt(encrypted_value.encode()).decode()
    except Exception:
        return encrypted_value # Return as-is if not encrypted
_chroma_client = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing ChromaDB at {CHROMA_DATA_PATH}...")
        os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    return _chroma_client


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """Standard chat payload from the Agent-fy Gateway."""
    message: str
    injected_context: dict = {}

class SyncRequest(BaseModel):
    """Payload for triggering a single inbox sync."""
    config: dict
    state: dict = {}

class KnowledgePayload(BaseModel):
    """Text-based knowledge ingestion payload."""
    tenant_id: str
    content: str
    metadata: dict = {}
    chunk_strategy: str = "paragraph"

class RAGConfigResponse(BaseModel):
    """Returned by /rag-config for gateway introspection."""
    rag_enabled: bool = True
    vector_store: str = "chromadb"
    embedding_model: str = "gemini-embedding-2"
    supported_files: list = [".pdf", ".docx", ".xlsx", ".txt"]
    chunk_strategy: str = "paragraph"
    max_file_size_mb: int = 10


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS & RAG
# ══════════════════════════════════════════════════════════════════════════════

def _get_genai_client() -> genai.Client | None:
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    if not api_key: return None
    http_options = types.HttpOptions(base_url=base_url) if base_url else None
    return genai.Client(api_key=api_key, http_options=http_options)

async def get_embeddings(text: str) -> list[float]:
    client = _get_genai_client()
    if not client: return []
    try:
        truncated = text[:8000] if len(text) > 8000 else text
        response = client.models.embed_content(model="gemini-embedding-2", contents=truncated)
        return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []

async def get_retrieved_context(tenant_id: str, query: str, n_results: int = 5) -> str:
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=f"tenant_{tenant_id}")
        if collection.count() == 0: return ""
        query_embedding = await get_embeddings(query)
        if not query_embedding: return ""
        results = collection.query(query_embeddings=[query_embedding], n_results=min(n_results, collection.count()))
        if not results["documents"] or not results["documents"][0]: return ""
        context_parts = []
        for i, doc_text in enumerate(results["documents"][0]):
            source = ""
            if results.get("metadatas") and results["metadatas"][0]:
                meta = results["metadatas"][0][i]
                source = meta.get("filename", meta.get("source", ""))
            prefix = f"[Source: {source}] " if source else ""
            context_parts.append(f"{prefix}{doc_text}")
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return ""

async def process_email_with_gemini(business_name: str, faq_text: str, message: str, tenant_id: Optional[str] = None) -> str:
    client = _get_genai_client()
    if not client: return "Sorry, I am currently unable to process your request."
    vector_context = await get_retrieved_context(tenant_id, message) if tenant_id else ""
    system_prompt = (
        f"You are a professional AI support assistant for **{business_name}**.\n\n"
        f"Use the following business information to answer the customer's inquiry:\n\n"
        f"── FAQ / Business Info ──\n{faq_text}\n\n"
    )
    if vector_context:
        system_prompt += f"── Retrieved Knowledge ──\n{vector_context}\n\n"
    system_prompt += (
        "── Guidelines ──\n"
        "• Be polite, professional, and concise.\n"
        "• Respond ONLY with a JSON object: {\"reply\": \"your response text\"}\n"
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, f"Customer email: \"{message}\""],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.5),
        )
        text_resp = response.text.strip()
        if text_resp.startswith("```json"): text_resp = text_resp[7:-3].strip()
        data = json.loads(text_resp)
        return data.get("reply", "Thank you for your message.")
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Thank you for your message. We will get back to you shortly."

# ══════════════════════════════════════════════════════════════════════════════
# SYNC LOGIC
# ══════════════════════════════════════════════════════════════════════════════

async def perform_sync(config: dict, state: dict) -> tuple[bool, dict, int]:
    """Performs a single sync of a Gmail inbox."""
    gmail_user = config.get("gmail_address", "").strip()
    gmail_pass = config.get("gmail_app_password", "").replace(" ", "").strip()
    business_name = config.get("business_name", "the business")
    faq_text = config.get("faq_text", "")
    tenant_id = config.get("tenant_id")
    last_uid = state.get("last_processed_uid")

    if not gmail_user or not gmail_pass:
        return True, state, 0

    new_state = state.copy()

    try:
        def _imap_sync():
            mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
            mail.login(gmail_user, gmail_pass)
            mail.select("INBOX")

            if last_uid is None:
                status, data = mail.uid("SEARCH", None, "ALL")
                uids = [int(u) for u in data[0].split()] if (status == "OK" and data[0]) else []
                current_last_uid = max(uids) if uids else 0
                mail.close()
                mail.logout()
                return current_last_uid, []

            status, messages = mail.uid("SEARCH", None, f"UNSEEN UID {last_uid + 1}:*")
            if status != "OK" or not messages[0]:
                mail.close()
                mail.logout()
                return None, []

            emails_data = []
            for e_uid in messages[0].split():
                status, msg_data = mail.uid("FETCH", e_uid, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        emails_data.append((int(e_uid), response_part[1]))
                mail.uid("STORE", e_uid, "+FLAGS", "\\Seen")

            mail.close()
            mail.logout()
            return None, emails_data

        last_uid_update, fetched_emails = await asyncio.to_thread(_imap_sync)
        
        if last_uid_update is not None:
            new_state["last_processed_uid"] = last_uid_update
            return True, new_state, 0

        max_processed_uid = last_uid
        for uid_int, raw_email in fetched_emails:
            msg = email.message_from_bytes(raw_email)
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

            if body.strip():
                reply_text = await process_email_with_gemini(business_name, faq_text, body.strip(), tenant_id=tenant_id)
                
                def _send_reply(r_text, r_sender, r_subject):
                    smtp_msg = MIMEText(r_text)
                    smtp_msg["Subject"] = f"Re: {r_subject}"
                    smtp_msg["From"] = gmail_user
                    smtp_msg["To"] = r_sender
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                        s.login(gmail_user, gmail_pass)
                        s.send_message(smtp_msg)

                await asyncio.to_thread(_send_reply, reply_text, sender, subject)
            
            max_processed_uid = max(max_processed_uid, uid_int)
        
        new_state["last_processed_uid"] = max_processed_uid
        return True, new_state, len(fetched_emails)

    except Exception as e:
        logger.error(f"Sync error for {gmail_user}: {e}")
        return False, state, 0


# ══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(content: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
    return "\n\n".join(pages)

def extract_text_from_docx(content: bytes) -> str:
    doc = docx.Document(io.BytesIO(content))
    return "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

def extract_text_from_xlsx(content: bytes) -> str:
    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    sheets = []
    for name in wb.sheetnames:
        rows = [" | ".join([str(c).strip() for c in r if c is not None]) for r in wb[name].iter_rows(values_only=True)]
        if rows: sheets.append(f"[Sheet: {name}]\n" + "\n".join(rows))
    return "\n\n".join(sheets)

def extract_text(content: bytes, filename: str) -> str:
    ext = filename.lower()
    if ext.endswith(".pdf"): return extract_text_from_pdf(content)
    if ext.endswith(".docx"): return extract_text_from_docx(content)
    if ext.endswith(".xlsx") or ext.endswith(".xls"): return extract_text_from_xlsx(content)
    return content.decode("utf-8", errors="replace")

def chunk_text(text: str, strategy: str = "paragraph", max_chunk_words: int = 400, overlap_words: int = 50) -> list[str]:
    """
    Split text into chunks for embedding.
    Strategies: paragraph, sentence, fixed_size
    """
    if not text or not text.strip():
        return []

    if strategy == "sentence":
        return _chunk_by_sentence(text, max_chunk_words, overlap_words)
    elif strategy == "fixed_size":
        return _chunk_fixed_size(text, max_chunk_words, overlap_words)
    else:
        return _chunk_by_paragraph(text, max_chunk_words, overlap_words)


def _chunk_by_paragraph(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split on paragraph boundaries, merge short paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())
        if para_words > max_words:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.extend(_chunk_fixed_size(para, max_words, overlap_words))
            continue

        if current_word_count + para_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            if overlap_words > 0:
                last_para = current_chunk[-1]
                current_chunk = [last_para]
                current_word_count = len(last_para.split())
            else:
                current_chunk = []
                current_word_count = 0

        current_chunk.append(para)
        current_word_count += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def _chunk_by_sentence(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split on sentence boundaries (. ! ?)."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_words + s_words > max_words and current:
            chunks.append(" ".join(current))
            overlap_text = " ".join(current)
            overlap_split = overlap_text.split()
            if overlap_words > 0 and len(overlap_split) > overlap_words:
                carry = " ".join(overlap_split[-overlap_words:])
                current = [carry]
                current_words = overlap_words
            else:
                current = []
                current_words = 0
        current.append(sentence)
        current_words += s_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def _chunk_fixed_size(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Fixed-size word windows with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        end = min(i + max_words, len(words))
        chunk = " ".join(words[i:end])
        if chunk.strip():
            chunks.append(chunk)
        i += max_words - overlap_words
        if i >= len(words):
            break
    return chunks

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"status": "healthy", "version": "3.0.0"}

@app.post("/sync")
async def handle_sync(payload: SyncRequest):
    success, updated_state, count = await perform_sync(payload.config, payload.state)
    return {"status": "success" if success else "error", "state": updated_state, "processed": count}

@app.post("/install")
async def handle_install():
    """Triggered by the Gateway as a wake-up signal."""
    return {"status": "ok", "message": "Agent woken up."}

@app.post("/")
async def handle_chat(payload: ChatRequest):
    config = payload.injected_context
    reply = await process_email_with_gemini(config.get("business_name", "the business"), config.get("faq_text", ""), payload.message, tenant_id=config.get("tenant_id"))
    return {"reply": reply}

@app.post("/index-knowledge")
async def index_knowledge(payload: KnowledgePayload):
    """
    Indexes text content into the tenant's ChromaDB collection.
    Applies smart chunking before embedding.
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=f"tenant_{payload.tenant_id}")

        # Chunk the content
        chunks = chunk_text(
            payload.content,
            strategy=payload.chunk_strategy,
            max_chunk_words=400,
            overlap_words=50,
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful text to index after chunking.")

        indexed_ids = []
        for i, chunk in enumerate(chunks):
            embedding = await get_embeddings(chunk)
            if not embedding:
                logger.warning(f"Skipping chunk {i} — embedding failed.")
                continue

            # Deterministic ID based on content hash for idempotency
            content_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
            doc_id = f"doc_{payload.tenant_id[:8]}_{content_hash}"

            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    **payload.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "indexed_at": time.time(),
                }],
            )
            indexed_ids.append(doc_id)

        logger.info(
            f"Indexed {len(indexed_ids)}/{len(chunks)} chunks for tenant {payload.tenant_id}"
        )
        return {
            "status": "success",
            "chunks_indexed": len(indexed_ids),
            "total_chunks": len(chunks),
            "doc_ids": indexed_ids,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-file")
async def upload_file(tenant_id: str = Form(...), file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text(content, file.filename)
    return await index_knowledge(KnowledgePayload(tenant_id=tenant_id, content=text))

@app.get("/rag-config", response_model=RAGConfigResponse)
async def rag_config():
    return RAGConfigResponse()


# ══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS BACKGROUND MONITORING
# ══════════════════════════════════════════════════════════════════════════════

async def autonomous_monitoring_loop():
    """
    Background task that periodically checks all active tenants assigned to this agent.
    """
    logger.info("Starting Autonomous Background Monitor...")
    agent_id = os.getenv("AGENT_ID", "email-support-agent")
    
    while True:
        try:
            # 1. Fetch all active tenants for THIS agent
            tenants_ref = db.collection("tenants").where("agent_id", "==", agent_id).where("status", "==", "active")
            docs = tenants_ref.stream()
            
            for doc in docs:
                tenant_id = doc.id
                data = doc.to_dict()
                
                # 2. Decrypt config
                config = data.get("config", {})
                decrypted_config = {}
                for key, value in config.items():
                    is_sensitive = any(word in key.lower() for word in ["password", "key", "token", "secret"])
                    if is_sensitive and isinstance(value, str):
                        decrypted_val = decrypt_value(value)
                        # Sanitize gmail app passwords
                        if "gmail_app_password" in key:
                            decrypted_val = decrypted_val.replace(" ", "").strip()
                        decrypted_config[key] = decrypted_val
                    else:
                        decrypted_config[key] = value
                
                decrypted_config["tenant_id"] = tenant_id
                
                # 3. Perform Sync
                logger.info(f"Background Sync: Checking inbox for tenant {tenant_id}...")
                current_state = data.get("state", {})
                
                # Run the same logic as the /sync endpoint
                success, new_state, processed_count = await perform_sync(decrypted_config, current_state)
                
                if success and (new_state != current_state or processed_count > 0):
                    # Update state in Firestore
                    db.collection("tenants").document(tenant_id).update({
                        "state": new_state,
                        "metrics.last_called_at": firestore.SERVER_TIMESTAMP
                    })
                    if processed_count > 0:
                        logger.info(f"Background Sync: Processed {processed_count} emails for {tenant_id}")
            
        except Exception as e:
            logger.error(f"Autonomous Monitor Error: {e}")
            
        # Poll interval (every 60 seconds)
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    # Start the background monitor when the app starts
    asyncio.create_task(autonomous_monitoring_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
