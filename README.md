# 🤖 Manan's RAG Chatbot: Decoupled FastAPI & Streamlit System

**Author:** Manan Mehta ([GitHub](https://github.com/MananMehta17) | [LinkedIn](https://www.linkedin.com/in/mananmehta08))

A high-performance, enterprise-grade, and decoupled Retrieval-Augmented Generation (RAG) chatbot designed as part of the Junior AI Engineer selection process at **Amlgo Labs**.

This system splits responsibilities into a high-performance **FastAPI backend** (which handles document parsing, sentence-aware chunking, local embedding generation, FAISS indexing, and LLM querying) and an interactive **Streamlit frontend** (which acts as a thin client). All sensitive credentials and configuration are managed securely on the backend via environment variables, offering zero-config access for end users.

---

## 🛠️ Tech Stack & Architecture

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) with asynchronous endpoint routing and event-streaming.
- **Frontend Thin Client**: [Streamlit](https://streamlit.io/) consuming real-time token line streams from the backend via HTTPX.
- **LLM Engine**: [Groq API](https://groq.com/) for lightning-fast inference on open-source instruction models (`llama-3.1-8b-instant` or similar).
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) for local flat-file index vector storage.
- **Embedding Generation**: Local [SentenceTransformers](https://www.sbert.net/) utilizing the `all-MiniLM-L6-v2` model (384-dimensional dense vectors generated locally on CPU).
- **Document Extractors**: `pypdf` for parsing PDF documents and plain text handlers for `.txt`/`.md` formats.

---

## 🚀 Setup & Installation Instructions

Follow these step-by-step instructions to get the application up and running on your local Windows system.

### 1. Initialize Virtual Environment (Recommended)
Open a terminal in the root of the project directory and create a virtual environment to isolate the project packages:

**In PowerShell / Command Prompt:**
```powershell
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment (PowerShell)
.venv\Scripts\Activate.ps1

# Activate the virtual environment (Command Prompt)
.venv\Scripts\activate.bat
```

### 2. Install Project Dependencies
With your virtual environment activated, run the following command to install all the required Python libraries:

```powershell
pip install -r requirements.txt
```

### 3. Configure the Environment Variables (`.env`)
Locate or create a file named `.env` in the root folder of the project. Open it and add your Groq API Key:

```env
# Root folder / .env
GROQ_API_KEY=your_actual_groq_api_key_here
DEFAULT_MODEL=llama-3.1-8b-instant
```
> **How to get a key?** Create a free account at [console.groq.com](https://console.groq.com/) and generate an API key.

### 4. Place Your Target Documents
Put any text or PDF documents that you want the chatbot to query inside the `/data` folder.
*Example: Place `sample_terms.pdf` or `corporate_policy.txt` into the `data/` directory.*

---

## 🏃 Running the Application

To run the decoupled RAG system, you need to run both the backend server and the frontend client concurrently. Open two separate terminal windows with the virtual environment activated:

### Terminal 1: Launch FastAPI Backend Server
Start the backend server. By default, it runs on port `8001` to avoid common port conflicts:

```powershell
# Activate env and run
python backend.py
```
*You should see uvicorn start successfully at: `http://127.0.0.1:8001`*

### Terminal 2: Launch Streamlit Frontend Client
Start the Streamlit client to open the user-friendly interface:

```powershell
# Activate env and run
streamlit run app.py
```
*The app will automatically open in your web browser at **`http://localhost:8501`**.*

---

## 💡 How to Test & Use the App

1. Once both services are running and the Streamlit interface is loaded, click the **🚀 Process & Index Documents** button in the sidebar. This will parse all files in `/data`, split them into sentence-aware chunks, generate embeddings locally, and build your FAISS index.
2. Once indexed, type your query in the chat box at the bottom.
3. Watch the answer stream in real-time, backed by expandable **Grounded Reference Sources** showing the exact document name, chunk index, and similarity scores used to generate the answer.
