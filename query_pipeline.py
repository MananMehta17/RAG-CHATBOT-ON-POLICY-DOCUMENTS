import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Generator, Tuple
from sentence_transformers import SentenceTransformer
from groq import Groq

class RAGPipeline:
    def __init__(self, 
                 vectordb_dir: str, 
                 chunks_manifest_path: str, 
                 embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the search and generation components.
        """
        self.vectordb_dir = vectordb_dir
        self.chunks_manifest_path = chunks_manifest_path
        self.embedding_model_name = embedding_model_name
        
        # Lazy load components when needed
        self._embedding_model = None
        self._index = None
        self._chunks = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            print(f"Loading embedding model '{self.embedding_model_name}'...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @property
    def index(self):
        if self._index is None:
            index_path = os.path.join(self.vectordb_dir, "index.faiss")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index file not found at: {index_path}. Please run ingestion and embedding first.")
            print(f"Loading FAISS index from {index_path}...")
            self._index = faiss.read_index(index_path)
        return self._index

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        if self._chunks is None:
            if not os.path.exists(self.chunks_manifest_path):
                raise FileNotFoundError(f"Chunks manifest file not found at: {self.chunks_manifest_path}")
            print(f"Loading chunks manifest from {self.chunks_manifest_path}...")
            with open(self.chunks_manifest_path, "r", encoding="utf-8") as f:
                self._chunks = json.load(f)
        return self._chunks

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Performs semantic search on the FAISS index and returns the top_k matching chunks.
        """
        # Generate query embedding
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_vector, top_k)
        
        retrieved_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk_data = self.chunks[idx].copy()
            # Append score (Euclidean distance; lower is closer)
            chunk_data["score"] = float(distances[0][i])
            retrieved_chunks.append(chunk_data)
            
        return retrieved_chunks

    def get_prompt_template(self, context: str, query: str) -> List[Dict[str, str]]:
        """
        Builds a high-quality, grounded system prompt template.
        """
        system_message = (
            "You are Manan's RAG Chatbot, a highly precise AI assistant designed to answer user queries "
            "strictly based on the provided document references. Refer to the facts directly.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Rely ONLY on the clear facts directly mentioned in the Context. Do not assume, extrapolate, or bring outside knowledge.\n"
            "2. If the answer cannot be found in the Context, explicitly state: 'Based on the provided references, I do not have enough information to answer your question.' Do NOT hallucinate or make up details.\n"
            "3. Cite your sources gracefully when answering. Keep your response direct, clear, and structured using markdown."
        )
        
        user_message = (
            f"Context references:\n"
            f"----------------------\n"
            f"{context}\n"
            f"----------------------\n\n"
            f"Query: {query}"
        )
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def generate_stream(self, 
                        query: str, 
                        groq_api_key: str, 
                        model_name: str = 'llama-3.1-8b-instant', 
                        top_k: int = 4) -> Generator[Dict[str, Any], None, None]:
        """
        Executes the RAG pipeline end-to-end and yields tokens/citations in a stream.
        """
        if not groq_api_key:
            yield {"type": "error", "content": "Groq API Key is required."}
            return
            
        try:
            # 1. Retrieve
            retrieved_chunks = self.retrieve(query, top_k=top_k)
            
            # Send citations first/alongside the response
            yield {"type": "citations", "content": retrieved_chunks}
            
            if not retrieved_chunks:
                yield {"type": "token", "content": "No relevant documents found. Please ingest files first."}
                return
                
            # 2. Format Context
            formatted_contexts = []
            for idx, chunk in enumerate(retrieved_chunks):
                formatted_contexts.append(
                    f"[Source: {chunk['source_file']} (Chunk {chunk['chunk_index']})]\n{chunk['text']}"
                )
            context_str = "\n\n".join(formatted_contexts)
            
            # 3. Create client
            client = Groq(api_key=groq_api_key)
            messages = self.get_prompt_template(context_str, query)
            
            # 4. Request streaming chat completion
            completion_stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0, # highly grounded temperature
                max_tokens=1024,
                stream=True
            )
            
            # 5. Stream response back
            for chunk in completion_stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield {"type": "token", "content": token}
                    
        except Exception as e:
            yield {"type": "error", "content": f"Pipeline Error: {str(e)}"}
            return
