import os
import re
import json
from typing import List, Dict, Any
from pypdf import PdfReader

def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing whitespace, removing typical page headers/footers pattern,
    and collapsing multiple spaces and newlines.
    """
    # Remove control characters and normalize newlines
    text = text.replace('\r', '\n')
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines into a single newline
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def parse_pdf(file_path: str) -> str:
    """
    Parses a PDF file and extracts raw text page by page.
    """
    print(f"Extracting text from PDF: {file_path}")
    reader = PdfReader(file_path)
    extracted_text = []
    
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            extracted_text.append(f"\n--- PAGE {page_num + 1} ---\n")
            extracted_text.append(page_text)
            
    return "".join(extracted_text)

def parse_txt(file_path: str) -> str:
    """
    Parses a plain text file.
    """
    print(f"Reading text file: {file_path}")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def sentence_aware_chunking(text: str, min_words: int = 100, max_words: int = 300) -> List[str]:
    """
    Splits text into chunks of 100-300 words using a sentence-aware approach.
    It groups complete sentences together until the word count threshold is reached.
    """
    # Simple regex to split text by sentences while preserving punctuation
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # If adding this sentence exceeds the max word count and we already have some content,
        # save the current chunk and start a new one.
        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
            
        # If the chunk is sufficiently large, we can finalize it early
        if current_word_count >= min_words and current_word_count <= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0
            
    # Append any remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def process_directory(data_dir: str, chunks_dir: str) -> List[Dict[str, Any]]:
    """
    Processes all PDF and TXT files in data_dir, chunks them, and saves the manifest.
    """
    os.makedirs(chunks_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created empty data directory: {data_dir}. Please place documents here.")
        return []
        
    all_chunks = []
    chunk_id = 0
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if not os.path.isfile(file_path):
            continue
            
        ext = os.path.splitext(filename)[1].lower()
        raw_text = ""
        
        try:
            if ext == '.pdf':
                raw_text = parse_pdf(file_path)
            elif ext in ['.txt', '.md']:
                raw_text = parse_txt(file_path)
            else:
                print(f"Skipping unsupported file: {filename}")
                continue
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            continue
            
        cleaned_text = clean_text(raw_text)
        file_chunks = sentence_aware_chunking(cleaned_text)
        
        print(f"Processed {filename}: generated {len(file_chunks)} chunks.")
        
        for idx, chunk_text in enumerate(file_chunks):
            all_chunks.append({
                "id": chunk_id,
                "source_file": filename,
                "chunk_index": idx,
                "text": chunk_text
            })
            chunk_id += 1
            
    if all_chunks:
        output_path = os.path.join(chunks_dir, "chunks_manifest.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_chunks)} chunks to {output_path}")
        
    return all_chunks

if __name__ == "__main__":
    # Test execution path
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    CHUNKS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chunks"))
    process_directory(DATA_PATH, CHUNKS_PATH)
