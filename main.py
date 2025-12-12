from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- Imports inside try-catch to prevent crash on startup ---
try:
    from pinecone import Pinecone
    from openai import OpenAI
except ImportError as e:
    print(f"CRITICAL ERROR: Missing libraries. {e}")

# --- Configuration ---
# אנחנו קוראים את המשתנים, אבל לא קורסים אם הם חסרים
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
LLMOD_BASE_URL = os.getenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "ted-rag"

# Model Params
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"
CHUNK_SIZE = 1000
OVERLAP_RATIO = 0.2
TOP_K = 3

app = FastAPI()

# --- Global Clients (Initialized lazily) ---
client = None
pc = None
index = None

def init_clients():
    """מנסה להתחבר רק כשצריך, כדי למנוע קריסה בהתחלה"""
    global client, pc, index
    
    # בדיקת מפתחות
    if not LLMOD_API_KEY:
        print("❌ Error: LLMOD_API_KEY is missing in Environment Variables!")
        return False
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY is missing in Environment Variables!")
        return False

    try:
        if client is None:
            print("🔄 Connecting to OpenAI/LLMod...")
            client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
        
        if pc is None:
            print("🔄 Connecting to Pinecone...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            
        return True
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

# --- Data Models ---
class QueryRequest(BaseModel):
    question: str

# --- Core Logic ---
def retrieve_context(query):
    if not init_clients():
        raise Exception("Server configuration error: check logs for missing keys.")

    try:
        xq = client.embeddings.create(input=query, model=EMBEDDING_MODEL).data[0].embedding
        res = index.query(vector=xq, top_k=TOP_K, include_metadata=True)
        
        contexts = []
        context_text_list = []
        for match in res['matches']:
            contexts.append({
                "talk_id": match['metadata'].get('talk_id', 'N/A'),
                "title": match['metadata'].get('title', 'N/A'),
                "chunk": match['metadata'].get('text', ''),
                "score": match['score']
            })
            context_text_list.append(match['metadata'].get('text', ''))
        return contexts, "\n\n---\n\n".join(context_text_list)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        # מחזיר רשימה ריקה במקום לקרוס
        return [], ""

# --- Endpoints ---

@app.get("/")
def read_root():
    # בדיקת בריאות - האם המפתחות קיימים?
    status = "OK"
    missing = []
    if not LLMOD_API_KEY: missing.append("LLMOD_API_KEY")
    if not PINECONE_API_KEY: missing.append("PINECONE_API_KEY")
    
    if missing:
        status = f"MISSING KEYS: {', '.join(missing)}"
    
    return {"status": "TED RAG System is Online", "config_check": status}

@app.get("/api/stats")
def get_stats():
    return {"chunk_size": CHUNK_SIZE, "overlap_ratio": OVERLAP_RATIO, "top_k": TOP_K}

@app.post("/api/prompt")
def ask_question(request: QueryRequest):
    question = request.question
    
    try:
        context_objects, context_str = retrieve_context(question)
    except Exception as e:
        # תופס שגיאות חיבור ומחזיר אותן כהודעה ברורה
        return {"response": f"System Error: {str(e)}. Please check Vercel Logs.", "context": []}
    
    if not context_str:
        return {"response": "I don't know based on the provided TED data.", "context": []}

    system_prompt = """You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages).
You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context.
If the answer cannot be determined from the provided context, respond: "I don't know based on the provided TED data."
Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful."""

    user_prompt = f"""Context information is below:
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
"""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=1.0
        )
        return {
            "response": response.choices[0].message.content,
            "context": context_objects,
            "Augmented_prompt": {"System": system_prompt, "User": user_prompt}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
