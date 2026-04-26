"""
Email Support Agent — Production RAG-powered email responder.

This agent:
  1. Monitors a Gmail inbox for new emails via IMAP polling
  2. Retrieves relevant context from a ChromaDB vector store (RAG)
  3. Generates professional replies via Gemini 2.5 Flash
  4. Sends replies via SMTP

Supports ingesting knowledge from: PDF, DOCX, XLSX, and plain text.
"""

import os
import io
import json
import asyncio
import time
import imaplib
import smtplib
import email
import hashlib
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
    version="2.0.0",
)

# ── Environment ──────────────────────────────────────────────────────────────
AGENT_ID = os.getenv("AGENT_ID", "default_agent")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── ChromaDB Initialization ─────────────────────────────────────────────────
# Use /tmp for writable ephemeral storage in Cloud Run
CHROMA_DATA_PATH = os.path.join("/tmp", "chroma_data")
_chroma_client = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing ChromaDB at {CHROMA_DATA_PATH}...")
        os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    return _chroma_client

# Polling state per Gmail account
STATE: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """Standard chat payload from the Agent-fy Gateway."""
    message: str
    injected_context: dict = {}

class KnowledgePayload(BaseModel):
    """Text-based knowledge ingestion payload."""
    tenant_id: str
    content: str
    metadata: dict = {}
    chunk_strategy: str = "paragraph"

class InstallRequest(BaseModel):
    """Webhook-style installation payload (non-managed-DB fallback)."""
    agent_id: str
    gmail_address: str
    gmail_app_password: str
    class Config:
        extra = "allow"

class RAGConfigResponse(BaseModel):
    """Returned by /rag-config for gateway introspection."""
    rag_enabled: bool = True
    vector_store: str = "chromadb"
    embedding_model: str = "gemini-embedding-2"
    supported_files: list = [".pdf", ".docx", ".xlsx", ".txt"]
    chunk_strategy: str = "paragraph"
    max_file_size_mb: int = 10


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def _get_genai_client() -> genai.Client | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is not set.")
        return None
    return genai.Client(api_key=api_key)


async def get_embeddings(text: str) -> list[float]:
    """Generate embeddings using gemini-embedding-2."""
    client = _get_genai_client()
    if not client:
        return []
    try:
        # Truncate very long text to avoid API limits
        truncated = text[:8000] if len(text) > 8000 else text
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=truncated,
        )
        return response.embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION — Multi-format
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from a PDF file."""
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(f"[Page {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def extract_text_from_docx(content: bytes) -> str:
    """Extract text from a DOCX file, preserving paragraph structure."""
    doc = docx.Document(io.BytesIO(content))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    return "\n\n".join(paragraphs)


def extract_text_from_xlsx(content: bytes) -> str:
    """Extract text from an XLSX file, converting each sheet to a readable format."""
    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    sheets = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            # Filter None values and convert to strings
            cells = [str(cell).strip() for cell in row if cell is not None]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            sheets.append(f"[Sheet: {sheet_name}]\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(sheets)


def extract_text(content: bytes, filename: str) -> str:
    """Route to the correct extractor based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(content)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        return extract_text_from_xlsx(content)
    else:
        # Fallback: treat as plain text
        return content.decode("utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════════════════
# SMART CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, strategy: str = "paragraph", max_chunk_words: int = 400, overlap_words: int = 50) -> list[str]:
    """
    Split text into chunks for embedding.

    Strategies:
      - paragraph: Split on double newlines, merge small paragraphs
      - sentence:  Split on sentence boundaries
      - fixed_size: Fixed word-count windows with overlap
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

        # If a single paragraph exceeds max, split it with fixed_size
        if para_words > max_words:
            # Flush current buffer first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            # Split the large paragraph
            chunks.extend(_chunk_fixed_size(para, max_words, overlap_words))
            continue

        if current_word_count + para_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Overlap: carry the last paragraph into the next chunk
            if overlap_words > 0 and current_chunk:
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
            # Overlap
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
# VECTOR RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

async def get_retrieved_context(tenant_id: str, query: str, n_results: int = 5) -> str:
    """Search the tenant's vector collection for relevant knowledge chunks."""
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=f"tenant_{tenant_id}")

        if collection.count() == 0:
            return ""

        query_embedding = await get_embeddings(query)
        if not query_embedding:
            return ""

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        # Build context with source attribution
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


# ══════════════════════════════════════════════════════════════════════════════
# LLM — Gemini 2.5 Flash
# ══════════════════════════════════════════════════════════════════════════════

async def process_email_with_gemini(
    business_name: str,
    faq_text: str,
    message: str,
    tenant_id: Optional[str] = None,
) -> str:
    """Generate a reply to a customer email using Gemini + RAG context."""
    client = _get_genai_client()
    if not client:
        return "Sorry, I am currently unable to process your request."

    # RAG retrieval
    vector_context = ""
    if tenant_id:
        vector_context = await get_retrieved_context(tenant_id, message)

    system_prompt = (
        f"You are a professional AI support assistant for **{business_name}**.\n\n"
        f"Use the following business information to answer the customer's inquiry:\n\n"
        f"── FAQ / Business Info ──\n{faq_text}\n\n"
    )

    if vector_context:
        system_prompt += (
            f"── Retrieved Knowledge (from business documents) ──\n"
            f"{vector_context}\n\n"
        )

    system_prompt += (
        "── Guidelines ──\n"
        "• Be polite, professional, and concise.\n"
        "• Ground your answers in the provided business information and retrieved knowledge.\n"
        "• If the information isn't available, acknowledge it and suggest contacting the business directly.\n"
        "• Do NOT fabricate information.\n"
        "• Respond ONLY with a JSON object: {\"reply\": \"your response text\"}\n"
    )

    user_message = f"Customer email: \"{message}\""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_prompt, user_message],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.5,
            ),
        )

        text_resp = response.text.strip()
        # Clean markdown code fences if present
        if text_resp.startswith("```json"):
            text_resp = text_resp[7:-3].strip()
        elif text_resp.startswith("```"):
            text_resp = text_resp[3:-3].strip()

        data = json.loads(text_resp)
        return data.get("reply", "Thank you for your message. We will get back to you shortly.")
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Thank you for your message. Our team will review it and get back to you."


# ══════════════════════════════════════════════════════════════════════════════
# EMAIL POLLING
# ══════════════════════════════════════════════════════════════════════════════

async def poll_once(config: dict) -> None:
    """Poll a single Gmail inbox for new emails and auto-reply."""
    gmail_user = config.get("gmail_address")
    gmail_pass = config.get("gmail_app_password")
    business_name = config.get("business_name", "the business")
    faq_text = config.get("faq_text", "")
    tenant_id = config.get("tenant_id")

    if not gmail_user or not gmail_pass:
        return

    if gmail_user not in STATE:
        STATE[gmail_user] = {"last_processed_uid": None}

    try:
        def _imap_polling():
            mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
            mail.login(gmail_user, gmail_pass)
            mail.select("INBOX")

            if STATE[gmail_user]["last_processed_uid"] is None:
                status, data = mail.uid("SEARCH", None, "ALL")
                uids = [int(u) for u in data[0].split()] if (status == "OK" and data[0]) else []
                last_uid = max(uids) if uids else 0
                mail.close()
                mail.logout()
                return last_uid, []

            search_criteria = f"UNSEEN UID {STATE[gmail_user]['last_processed_uid'] + 1}:*"
            status, messages = mail.uid("SEARCH", None, search_criteria)
            
            if status != "OK" or not messages[0]:
                mail.close()
                mail.logout()
                return None, []

            found_uids = messages[0].split()
            emails_data = []
            for e_uid in found_uids:
                status, msg_data = mail.uid("FETCH", e_uid, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        emails_data.append((int(e_uid), response_part[1]))
                
                # Mark as seen immediately after fetching to avoid re-processing if later steps fail
                mail.uid("STORE", e_uid, "+FLAGS", "\\Seen")

            mail.close()
            mail.logout()
            return None, emails_data

        last_uid_update, fetched_emails = await asyncio.to_thread(_imap_polling)
        
        if last_uid_update is not None:
            STATE[gmail_user]["last_processed_uid"] = last_uid_update
            return

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

            if not body.strip():
                continue

            reply_text = await process_email_with_gemini(
                business_name, faq_text, body.strip(), tenant_id=tenant_id
            )

            def _send_reply(r_text, r_sender, r_subject):
                smtp_msg = MIMEText(r_text)
                smtp_msg["Subject"] = f"Re: {r_subject}"
                smtp_msg["From"] = gmail_user
                smtp_msg["To"] = r_sender

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtpserver:
                    smtpserver.login(gmail_user, gmail_pass)
                    smtpserver.send_message(smtp_msg)

            await asyncio.to_thread(_send_reply, reply_text, sender, subject)

            STATE[gmail_user]["last_processed_uid"] = max(
                STATE[gmail_user]["last_processed_uid"], uid_int
            )

    except Exception as e:
        logger.error(f"Polling error for {gmail_user}: {e}")


async def agent_polling_loop() -> None:
    """Background loop that polls all tenant inboxes."""
    logger.info("Starting Email Support Agent polling loop...")
    while True:
        await asyncio.sleep(10)
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(
                    f"{GATEWAY_URL}/api/developers/{AGENT_ID}/tenants",
                    timeout=5.0,
                )
                if res.status_code == 200:
                    tenants = res.json().get("tenants", [])
                    for tenant in tenants:
                        await poll_once(tenant)
        except Exception:
            pass  # Keep running quietly even if gateway is temporarily down


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Health check endpoint for Cloud Run."""
    return {
        "status": "Email Support Agent is healthy",
        "agent_id": AGENT_ID,
        "version": "2.0.0"
    }


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(agent_polling_loop())


@app.get("/rag-config", response_model=RAGConfigResponse)
async def rag_config():
    """Returns the RAG capabilities of this agent (used by gateway introspection)."""
    return RAGConfigResponse()


@app.post("/install")
async def handle_install(payload: InstallRequest):
    """Fallback route for developers who do NOT use managed DB."""
    data = payload.dict()
    logger.info(f"Received manual installation for {data.get('gmail_address')}")
    return {"status": "installed"}


@app.post("/")
async def handle_chat(payload: ChatRequest):
    """
    Standard chat endpoint called by the Agent-fy Gateway.
    Uses the injected_context (business config) to generate responses.
    """
    logger.info(f"Chat request: {payload.message[:80]}...")
    config = payload.injected_context
    business_name = config.get("business_name", config.get("restaurant_name", "the business"))
    faq_text = config.get("faq_text", "")
    tenant_id = config.get("tenant_id")

    reply = await process_email_with_gemini(business_name, faq_text, payload.message, tenant_id=tenant_id)
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
async def upload_file(
    tenant_id: str = Form(...),
    file: UploadFile = File(...),
    chunk_strategy: str = Form("paragraph"),
):
    """
    Accepts a raw file upload, extracts text, chunks it, and indexes into ChromaDB.
    Supports: PDF, DOCX, XLSX, TXT.
    """
    # Validate file extension
    supported = {".pdf", ".docx", ".xlsx", ".xls", ".txt"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(supported))}",
        )

    # Read and validate size (10 MB limit)
    content = await file.read()
    max_size = 10 * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Max: 10 MB.",
        )

    # Extract text
    try:
        text = extract_text(content, file.filename or "file.txt")
    except Exception as e:
        logger.error(f"Text extraction failed for {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from file: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the file.")

    # Chunk and index
    chunks = chunk_text(text, strategy=chunk_strategy, max_chunk_words=400, overlap_words=50)

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=f"tenant_{tenant_id}")
    indexed = 0

    for i, chunk in enumerate(chunks):
        embedding = await get_embeddings(chunk)
        if not embedding:
            continue

        content_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
        doc_id = f"file_{tenant_id[:8]}_{content_hash}"

        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "filename": file.filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_type": ext,
                "indexed_at": time.time(),
            }],
        )
        indexed += 1

    logger.info(f"File '{file.filename}' — indexed {indexed}/{len(chunks)} chunks for tenant {tenant_id}")
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_indexed": indexed,
        "total_chunks": len(chunks),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
