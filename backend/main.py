"""
Groq + HuggingFace + Chroma + Firestore Observability Platform
---------------------------------------------------------------
✅ Semantic Splitter Before Parsing (prevent LLM overflow)
✅ Semantic Splitter Before Embedding (improve context precision)
✅ Firestore for tenant/app isolation
✅ Chroma Vector DB for RAG
"""

from fastapi import FastAPI, Body
from google.cloud import firestore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
# from langchain.text_splitter import SemanticChunker
from langchain_experimental.text_splitter import SemanticChunker
from pydantic import BaseModel
from typing import List, Optional
import os, uuid, datetime, json, logging
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

# ---------------------------------------------------------
# FASTAPI + CORS CONFIG
# ---------------------------------------------------------
app = FastAPI(title="Groq + HuggingFace Observability AI")

# ✅ Allow your React frontend (localhost:3000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, OPTIONS, DELETE, etc.
    allow_headers=["*"],   # Content-Type, Authorization, etc.
)


# ======================================================
# ENVIRONMENT SETUP
# ======================================================
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

app = FastAPI(title="Groq + HuggingFace Observability AI (Semantic Splitters)")

# Firestore setup
db = firestore.Client()

# ======================================================
# LLM + EMBEDDING SETUP
# ======================================================
groq_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
    temperature=0.2,
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")

# ======================================================
# PYDANTIC SCHEMAS
# ======================================================
class ParsedLog(BaseModel):
    timestamp: Optional[str]
    log_level: Optional[str]
    service_name: Optional[str]
    error_code: Optional[str]
    message: str

class LogBatch(BaseModel):
    logs: List[ParsedLog]

class AnomalyReport(BaseModel):
    summary: str
    detected_anomalies: List[str]
    probable_causes: List[str]
    severity_level: str
    recommended_action: str

# ======================================================
# HELPERS
# ======================================================
def semantic_split_text(raw_text: str, stage: str):
    """Split logs semantically into coherent chunks."""
    splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile"
    )
    chunks = splitter.split_text(raw_text)
    logging.info(f"🧠 Semantic splitter ({stage}) created {len(chunks)} chunks.")
    return chunks


def parse_logs_via_llm(raw_logs: str) -> List[ParsedLog]:
    """Parse raw logs into structured JSON."""
    logging.info("🧩 Step 1: Parsing logs via Groq LLM...")
    parser = PydanticOutputParser(pydantic_object=LogBatch)
    prompt = ChatPromptTemplate.from_template("""
    You are a log parser AI. Extract structured fields:
    timestamp, log_level, service_name, error_code, message.
    Return valid JSON matching:
    {format_instructions}
    Logs:
    {logs}
    """)

    formatted = prompt.format_prompt(
        logs=raw_logs,
        format_instructions=parser.get_format_instructions()
    )
    response = groq_llm.invoke(formatted.to_string())

    try:
        parsed = parser.parse(response.content)
        logging.info(f"✅ Parsed {len(parsed.logs)} logs successfully.")
        return parsed.logs
    except Exception as e:
        logging.error(f"❌ Parsing failed: {e}")
        logging.debug(response.content)
        raise


def parse_large_logs_via_llm(raw_logs: str):
    """Handles large log blobs safely using semantic chunking before parsing."""
    chunks = semantic_split_text(raw_logs, stage="pre-parsing")
    all_parsed_logs = []

    for i, chunk in enumerate(chunks, start=1):
        logging.info(f"🔹 Parsing semantic chunk {i}/{len(chunks)}...")
        try:
            parsed_chunk = parse_logs_via_llm(chunk)
            all_parsed_logs.extend(parsed_chunk)
        except Exception as e:
            logging.error(f"❌ Failed parsing chunk {i}: {e}")

    logging.info(f"✅ Total structured logs parsed: {len(all_parsed_logs)}")
    return all_parsed_logs


def get_chroma_collection(tenant_id, app_id):
    """Get or create Chroma collection for a tenant/app."""
    name = f"{tenant_id}_{app_id}"
    logging.info(f"🗂️ Initializing Chroma collection: {name}")
    return Chroma(collection_name=name, embedding_function=embedding_model, persist_directory=CHROMA_PATH)


def build_vector_store(logs, tenant_id, app_id):
    """Semantic splitter before embeddings for contextual precision."""
    logging.info("🔢 Step 2: Building semantic vector store...")

    # Convert structured logs into plain text
    log_text = "\n".join([
        f"[{l.timestamp or 'N/A'}] {l.service_name or ''} - {l.message}" for l in logs
    ])

    # Semantic splitting before embeddings
    semantic_chunks = semantic_split_text(log_text, stage="pre-embedding")

    store = get_chroma_collection(tenant_id, app_id)
    store.add_texts(semantic_chunks)
    store.persist()
    logging.info(f"✅ Stored {len(semantic_chunks)} semantic chunks in Chroma ({tenant_id}_{app_id}).")
    return store


def store_logs_firestore(tenant_id, app_id, logs):
    """Store structured logs in Firestore by tenant/app."""
    logging.info("💾 Step 3: Storing logs in Firestore...")
    batch = db.batch()
    for log in logs:
        ref = db.collection("tenants").document(tenant_id).collection("apps").document(app_id).collection("logs").document(str(uuid.uuid4()))
        batch.set(ref, log.dict())
    batch.commit()
    logging.info(f"✅ Stored {len(logs)} logs in Firestore.")


def detect_anomalies_via_llm(logs) -> dict:
    """Detect anomalies using Groq LLM."""
    logging.info("🧠 Step 4: Detecting anomalies via Groq LLM...")
    parser = PydanticOutputParser(pydantic_object=AnomalyReport)
    context = "\n".join([f"[{l.timestamp}] {l.message}" for l in logs[:50]])

    prompt = ChatPromptTemplate.from_template("""
    You are an observability AI. Analyze these logs and detect anomalies.
    Return structured JSON following:
    {format_instructions}
    Logs:
    {context}
    """)

    formatted = prompt.format_prompt(
        context=context,
        format_instructions=parser.get_format_instructions()
    )

    response = groq_llm.invoke(formatted.to_string())
    try:
        report = parser.parse(response.content)
        logging.info("✅ Anomaly detection complete.")
        return report.dict()
    except Exception:
        logging.warning("⚠️ Could not parse anomaly output.")
        return {"summary": response.content}


def store_anomalies_firestore(tenant_id, app_id, summary):
    """Save anomaly reports per tenant/app in Firestore."""
    logging.info("💾 Step 5: Saving anomaly summary...")
    doc_id = f"{tenant_id}_{app_id}_{uuid.uuid4().hex}"

    db.collection("tenants").document(tenant_id).collection("apps").document(app_id).collection("anomalies").document(doc_id).set({
        "tenant_id": tenant_id,
        "app_id": app_id,
        "created_at": datetime.datetime.utcnow().isoformat(),
        **summary,
    })
    logging.info(f"✅ Anomaly summary stored in Firestore for {tenant_id}/{app_id}.")

# ======================================================
# 🔹 Conversational Memory Helpers (Client-Managed convo_id)
# ======================================================

def get_conversation_ref(username, tenant_id, app_id, convo_id):
    """Return a Firestore reference to the conversation document."""
    return (
        db.collection("user_collections")
        .document(username)
        .collection("tenants")
        .document(tenant_id)
        .collection("apps")
        .document(app_id)
        .collection("conversations")
        .document(convo_id)
    )


def ensure_conversation_exists(username, tenant_id, app_id, convo_id):
    """Ensure conversation document exists in Firestore."""
    ref = get_conversation_ref(username, tenant_id, app_id, convo_id)
    if not ref.get().exists:
        ref.set({
            "created_at": datetime.datetime.utcnow().isoformat(),
            "summary": "",
            "token_count": 0,
        })
        logging.info(f"🆕 Created new conversation {convo_id} for {username}/{tenant_id}/{app_id}")
    return ref


def add_message_to_conversation(username, tenant_id, app_id, convo_id, role, content):
    convo_ref = (
        db.collection("user_conversations")
        .document(username)
        .collection("tenants")
        .document(tenant_id)
        .collection("apps")
        .document(app_id)
        .collection("conversations")
        .document(convo_id)
        .collection("messages")
    )

    convo_ref.add({
        "role": role,  # "user" or "assistant"
        "content": content,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })



def fetch_conversation_history(username, tenant_id, app_id, convo_id, limit=20):
    """Fetch recent messages (most recent last)."""
    ref = (
        get_conversation_ref(username, tenant_id, app_id, convo_id)
        .collection("messages")
    )
    docs = ref.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit).stream()
    return [doc.to_dict() for doc in reversed(list(docs))]


def summarize_conversation_if_needed(username, tenant_id, app_id, convo_id, token_limit=5000):
    """If conversation exceeds token limit, summarize older messages."""
    convo_ref = get_conversation_ref(username, tenant_id, app_id, convo_id)
    messages = fetch_conversation_history(username, tenant_id, app_id, convo_id, limit=50)

    full_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    token_estimate = len(full_text.split())

    if token_estimate < token_limit:
        return None

    logging.info(f"🧠 Token limit exceeded ({token_estimate}). Summarizing...")

    prompt = ChatPromptTemplate.from_template("""
    Summarize the following chat preserving anomalies, reasoning, and key context.
    Keep summary concise (~400 tokens max).
    Chat history:
    {history}
    """)
    formatted = prompt.format_prompt(history=full_text)
    summary_response = groq_llm.invoke(formatted.to_string())
    summary_text = summary_response.content.strip()

    convo_ref.set({"summary": summary_text, "token_count": token_estimate}, merge=True)

    # Keep only last 4 messages
    msgs_ref = convo_ref.collection("messages")
    all_msgs = list(msgs_ref.order_by("created_at").stream())
    keep = {doc.id for doc in all_msgs[-4:]}
    for doc in all_msgs:
        if doc.id not in keep:
            doc.reference.delete()

    logging.info(f"✅ Summarized conversation {convo_id} for {username} and retained last 4 messages.")
    return summary_text

# ======================================================
# 🔍 RECENT DATA HELPERS
# ======================================================
def get_recent_logs_from_firestore(tenant_id: str, app_id: str, limit: int = 10):
    """Fetch latest structured logs from Firestore."""
    try:
        logs_ref = (
            db.collection("tenants")
              .document(tenant_id)
              .collection("apps")
              .document(app_id)
              .collection("logs")
              .order_by("timestamp", direction=firestore.Query.DESCENDING)
              .limit(limit)
        )
        docs = logs_ref.stream()
        logs = [doc.to_dict() for doc in docs]
        return logs
    except Exception as e:
        logging.error(f"❌ Failed to fetch logs: {e}")
        return []


def get_recent_anomalies_from_firestore(tenant_id: str, app_id: str, limit: int = 5):
    """Fetch latest anomaly summaries from Firestore."""
    try:
        anomalies_ref = (
            db.collection("tenants")
              .document(tenant_id)
              .collection("apps")
              .document(app_id)
              .collection("anomalies")
              .order_by("created_at", direction=firestore.Query.DESCENDING)
              .limit(limit)
        )
        docs = anomalies_ref.stream()
        anomalies = [doc.to_dict() for doc in docs]
        return anomalies
    except Exception as e:
        logging.error(f"❌ Failed to fetch anomalies: {e}")
        return []


# ======================================================
# 📊 DASHBOARD DATA ENDPOINTS
# ======================================================

@app.get("/recent_logs")
def recent_logs(tenant_id: str, app_id: str, limit: int = 10):
    """
    Fetch recent structured logs for a specific tenant/app.
    Returns the latest N logs sorted by timestamp (descending).
    """
    logging.info(f"📥 Fetching recent logs for {tenant_id}/{app_id} (limit={limit})...")
    logs = get_recent_logs_from_firestore(tenant_id, app_id, limit)
    return {
        "tenant_id": tenant_id,
        "app_id": app_id,
        "count": len(logs),
        "logs": logs
    }


@app.get("/recent_anomalies")
def recent_anomalies(tenant_id: str, app_id: str, limit: int = 5):
    """
    Fetch recent anomaly summaries for a specific tenant/app.
    Returns the latest N anomaly reports sorted by creation time.
    """
    logging.info(f"📥 Fetching recent anomalies for {tenant_id}/{app_id} (limit={limit})...")
    anomalies = get_recent_anomalies_from_firestore(tenant_id, app_id, limit)
    return {
        "tenant_id": tenant_id,
        "app_id": app_id,
        "count": len(anomalies),
        "anomalies": anomalies
    }


# ======================================================
# ENDPOINTS
# ======================================================
@app.post("/fetch_and_store_logs")
def fetch_and_store_logs(tenant_id: str, app_id: str):
    """Generate synthetic logs using Groq, parse, store, embed, and analyze."""
    logging.info(f"🚀 Generating synthetic logs for {tenant_id}/{app_id}...")
    prompt = ChatPromptTemplate.from_template("""
    Generate 10 realistic logs for tenant {tenant_id} and app {app_id}.
    Include a mix of INFO, WARN, and ERROR entries with timestamps.
    """)

    formatted = prompt.format_prompt(tenant_id=tenant_id, app_id=app_id)
    response = groq_llm.invoke(formatted.to_string())
    raw_logs = response.content.strip()
    logging.info("✅ Synthetic logs generated successfully.")
    return store_logs(tenant_id=tenant_id, app_id=app_id, raw_logs=raw_logs)


@app.post("/store_logs")
def store_logs(tenant_id: str , app_id: str, raw_logs: str = Body(...)):
    """End-to-end log processing pipeline."""
    logging.info(f"🚀 Received logs for tenant={tenant_id}, app={app_id}")

    structured = parse_large_logs_via_llm(raw_logs)
    store_logs_firestore(tenant_id, app_id, structured)
    build_vector_store(structured, tenant_id, app_id)
    anomaly = detect_anomalies_via_llm(structured)
    store_anomalies_firestore(tenant_id, app_id, anomaly)

    return {
        "tenant_id": tenant_id,
        "app_id": app_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "logs_stored": len(structured),
        "anomaly_summary": anomaly,
    }

@app.post("/ask_ai")
def ask_ai(
    username: str = Body(...),
    tenant_id: str = Body(...),
    app_id: str = Body(...),
    convo_id: str = Body(...),
    question: str = Body(...)
):
    """
    Context-aware conversational RAG endpoint.
    Client provides convo_id (e.g., 'convo_1').
    System maintains summary + message history per user.
    """
    logging.info(f"💬 Query from {username} for convo={convo_id}")

    # Ensure conversation exists
    ensure_conversation_exists(username, tenant_id, app_id, convo_id)

    # Step 1: Fetch logs retriever
    store = get_chroma_collection(tenant_id, app_id)
    retriever = store.as_retriever(search_kwargs={"k": 8})

    # Step 2: Fetch conversation history and summary
    convo_ref = get_conversation_ref(username, tenant_id, app_id, convo_id)
    convo_doc = convo_ref.get().to_dict() or {}
    summary = convo_doc.get("summary", "")
    history = fetch_conversation_history(username, tenant_id, app_id, convo_id)
    summarize_conversation_if_needed(username, tenant_id, app_id, convo_id)

    # Step 3: Combine history + summary into context
    recent_context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])
    context = f"Summary:\n{summary}\n\nRecent messages:\n{recent_context}"

    # Step 4: Build QA chain
    prompt = ChatPromptTemplate.from_template("""
    You are an observability assistant.
    Use context and retrieved log data to answer the user precisely.
    The logs of the below format JSON:
    summary, anomalies, probable_cause, recommendations.
                                   
    Context:
    {context}
    Question:
    {question}
                                              
    Just give me the answer or response to this questionin string format.
    """)

    chain = RetrievalQA.from_chain_type(
        llm=groq_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # response = chain.run({"context": context, "question": question})

    try:
        response = chain.invoke({"query": question, "context": context})
        result_text = response.get("result", "No answer generated.")
        logging.info(response)
    except Exception as e:
        logging.error(f"❌ LLM call failed: {e}")
        result_text = f"Error: {str(e)}"

    # Step 5: Store messages
    add_message_to_conversation(username, tenant_id, app_id, convo_id, "user", question)
    add_message_to_conversation(username, tenant_id, app_id, convo_id, "assistant", response)

    # Step 6: Return structured response
    try:
        result = json.loads(response)
    except Exception:
        result = {"summary": response}

    result.update({
        "username": username,
        "tenant_id": tenant_id,
        "app_id": app_id,
        "convo_id": convo_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })

    logging.info(f"✅ Responded to {username} for convo={convo_id}")
    logging.info(result)
    return result

from fastapi.responses import JSONResponse

@app.options("/ask_ai")
async def ask_ai_options():
    """Handles CORS preflight requests for /ask_ai."""
    headers = {
        "Access-Control-Allow-Origin": "http://localhost:3000",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }
    return JSONResponse(content={"status": "ok"}, headers=headers)


@app.get("/")
def root():
    return {"status": "running", "message": "Groq + HuggingFace Observability AI Active (Semantic Splitters)"}
