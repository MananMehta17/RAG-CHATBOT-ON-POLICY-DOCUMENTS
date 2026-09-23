import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_index(chunks_manifest_path: str, vectordb_dir: str, model_name: str = 'all-MiniLM-L6-v2') -> bool:
    """
    Loads chunk texts, generates embeddings using sentence-transformers,
    creates a FAISS index, and saves it.
    """
    if not os.path.exists(chunks_manifest_path):
        print(f"Chunks manifest not found: {chunks_manifest_path}. Please run ingestion first.")
        return False
        
    os.makedirs(vectordb_dir, exist_ok=True)
    
    print(f"Loading chunks manifest from {chunks_manifest_path}...")
    with open(chunks_manifest_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    if not chunks:
        print("Manifest is empty. No chunks to index.")
        return False
        
    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    print("Extracting text and generating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Check embedding dimension
    dimension = embeddings.shape[1]
    print(f"Generated {embeddings.shape[0]} embeddings of size {dimension}.")
    
    # Initialize FAISS IndexFlatL2 (standard L2 distance for similarity search)
    index = faiss.IndexFlatL2(dimension)
    
    # FAISS expects float32 arrays
    embeddings_f32 = embeddings.astype('float32')
    index.add(embeddings_f32)
    
    # Save the index to file
    index_path = os.path.join(vectordb_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"Successfully saved FAISS index to {index_path}")
    
    return True

if __name__ == "__main__":
    # Test execution path
    CHUNKS_MANIFEST = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chunks/chunks_manifest.json"))
    VECTORDB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vectordb"))
    build_index(CHUNKS_MANIFEST, VECTORDB_PATH)
