import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.ingest import process_directory
from src.embed import build_index
from src.query_pipeline import RAGPipeline

# Load environment configuration
load_dotenv()

app = FastAPI(title="Manan's RAG Backend")

# Read settings from environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "llama-3.1-8b-instant")

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
VECTORDB_DIR = os.path.join(BASE_DIR, "vectordb")
CHUNKS_MANIFEST = os.path.join(CHUNKS_DIR, "chunks_manifest.json")

# Ensure required directories exist
for d in [DATA_DIR, CHUNKS_DIR, VECTORDB_DIR]:
    os.makedirs(d, exist_ok=True)

# Initialize global Pipeline (lazy-loaded inside classes)
pipeline = RAGPipeline(
    vectordb_dir=VECTORDB_DIR,
    chunks_manifest_path=CHUNKS_MANIFEST
)

class QueryRequest(BaseModel):
    query: str

@app.post("/ingest")
def trigger_ingest():
    """
    Ingests all files inside the /data directory, chunks them, and builds a FAISS vector index.
    """
    try:
        files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f)) and f.lower().endswith(('.txt', '.pdf', '.md'))]
        if not files:
            raise HTTPException(status_code=400, detail="Please place at least one .txt or .pdf file inside the /data directory first.")
        
        chunks = process_directory(DATA_DIR, CHUNKS_DIR)
        success = build_index(CHUNKS_MANIFEST, VECTORDB_DIR)
        
        if success:
            # Force reloading the index and chunks on the pipeline instance
            pipeline._chunks = None
            pipeline._index = None
            return {
                "status": "success",
                "message": "Documents successfully ingested and indexed.",
                "chunks_count": len(chunks)
            }
        else:
            raise HTTPException(status_code=500, detail="Indexing and embedding generation failed.")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def get_status():
    """
    Reports current configuration, indexing status, and uploaded source files.
    """
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f)) and f.lower().endswith(('.txt', '.pdf', '.md'))]
    
    num_chunks = 0
    if os.path.exists(CHUNKS_MANIFEST):
        try:
            with open(CHUNKS_MANIFEST, "r", encoding="utf-8") as f:
                num_chunks = len(json.load(f))
        except Exception:
            pass
            
    index_exists = os.path.exists(os.path.join(VECTORDB_DIR, "index.faiss"))
    has_api_key = bool(GROQ_API_KEY)
    
    return {
        "model_in_use": DEFAULT_MODEL,
        "indexed_chunks": num_chunks,
        "faiss_index_active": index_exists,
        "has_api_key": has_api_key,
        "source_files": files
    }

@app.post("/query")
def query_rag(request: QueryRequest):
    """
    Queries the RAG pipeline and returns a real-time JSON-line chunk stream.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="Groq API Key is not set on the server environment. Please configure it in .env file.")
        
    index_path = os.path.join(VECTORDB_DIR, "index.faiss")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=400, detail="Vector index not found. Please click 'Process & Index Documents' first.")
        
    def event_stream():
        try:
            stream = pipeline.generate_stream(
                query=request.query,
                groq_api_key=GROQ_API_KEY,
                model_name=DEFAULT_MODEL
            )
            for chunk in stream:
                yield json.dumps(chunk) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"
            
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    # Default model configuration can be checked via status
    print(f"Starting server with default model: {DEFAULT_MODEL}")
    uvicorn.run(app, host="127.0.0.1", port=8001)
