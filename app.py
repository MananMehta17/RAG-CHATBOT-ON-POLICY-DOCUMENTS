import os
import json
import streamlit as st
import httpx

# Set Page Config with custom layout and theme aesthetics
st.set_page_config(
    page_title="Manan's RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend service configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001")

# Custom premium styling using CSS
st.markdown("""
<style>
    /* Premium Font and Color Styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #FF6B6B, #4D96FF, #6BCB77);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Elegant Custom Card for Source Citations */
    .source-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #4D96FF;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    .source-header {
        font-weight: 600;
        color: #4D96FF;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    
    .source-body {
        font-size: 0.85rem;
        color: #E2E8F0;
        line-height: 1.5;
    }
    
    .source-score {
        float: right;
        font-size: 0.8rem;
        color: #A0AEC0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Fetch current system status from FastAPI backend
@st.cache_data(ttl=2)
def get_backend_status():
    try:
        response = httpx.get(f"{BACKEND_URL}/status", timeout=3.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

status_data = get_backend_status()

# Header Section
st.markdown("<h1 class='main-title'>Manan's RAG</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Decoupled Architecture: FastAPI Backend & Streamlit Frontend</p>", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar setup
with st.sidebar:
    st.markdown("### ⚙️ System Control Panel")
    
    if status_data is None:
        st.error("🔴 Backend Server is offline. Please start backend.py first.")
    else:
        st.success("🟢 Backend Server is active.")
        
    st.markdown("---")
    st.markdown("### 📁 Data Ingestion Pipeline")
    
    if status_data:
        source_files = status_data.get("source_files", [])
        st.write(f"**Found files in `/data`:** {len(source_files)}")
        for f in source_files[:5]:
            st.caption(f"📄 {f}")
        if len(source_files) > 5:
            st.caption(f"... and {len(source_files) - 5} more files")
            
        # Ingest button
        if st.button("🚀 Process & Index Documents", use_container_width=True):
            if not source_files:
                st.error("Please place at least one .txt or .pdf file inside the `/data` directory first!")
            else:
                with st.status("Preprocessing documents and building FAISS index...", expanded=True) as status:
                    try:
                        response = httpx.post(f"{BACKEND_URL}/ingest", timeout=60.0)
                        if response.status_code == 200:
                            res_json = response.json()
                            status.update(label="Ingestion complete! Index ready.", state="complete", expanded=False)
                            st.success(f"Successfully indexed {res_json.get('chunks_count', 0)} chunks.")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            status.update(label="Embedding step failed.", state="error")
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        status.update(label="Communication error.", state="error")
                        st.error(f"Could not reach backend: {str(e)}")
    else:
        st.warning("Connect to the FastAPI server to trigger ingestion.")
                    
    st.markdown("---")
    st.markdown("### 📊 Status & Statistics")
    
    if status_data:
        st.info(f"""
        **Current LLM**: `{status_data.get('model_in_use', 'N/A')}`
        **Indexed Chunks**: `{status_data.get('indexed_chunks', 0)}`
        **FAISS Index Status**: `{"Active 🟢" if status_data.get('faiss_index_active') else "Empty 🔴"}`
        **API Configured**: `{"Yes 🟢" if status_data.get('has_api_key') else "Missing 🔴"}`
        """)
    else:
        st.info("System statistics unavailable while backend is offline.")
        
    # Reset Chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()

# Display Chat History
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        
        # Display Citations if assistant message has them
        if chat["role"] == "assistant" and "citations" in chat and chat["citations"]:
            with st.expander("📚 View Grounded Reference Sources"):
                for cit in chat["citations"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-header">
                            📄 {cit['source_file']} &nbsp;|&nbsp; Chunk {cit['chunk_index']}
                            <span class="source-score">L2 Distance: {cit['score']:.4f}</span>
                        </div>
                        <div class="source-body">
                            {cit['text']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Main query block
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    if status_data is None:
        st.error("Cannot query. Backend Server is offline.")
    else:
        # Display user query
        with st.chat_message("user"):
            st.markdown(prompt)
            
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Process assistant response with streaming
        with st.chat_message("assistant"):
            response_text = ""
            citations = []
            
            message_placeholder = st.empty()
            
            # Post request with streaming response
            try:
                with st.spinner("Retrieving references and generating response..."):
                    with httpx.stream("POST", f"{BACKEND_URL}/query", json={"query": prompt}, timeout=60.0) as r:
                        if r.status_code == 400:
                            # Read error detail from standard response
                            err_msg = json.loads(r.read().decode())["detail"]
                            message_placeholder.error(err_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
                        elif r.status_code != 200:
                            err_msg = f"Backend Server Error: Status code {r.status_code}"
                            message_placeholder.error(err_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
                        else:
                            for line in r.iter_lines():
                                if not line:
                                    continue
                                chunk = json.loads(line)
                                if chunk["type"] == "citations":
                                    citations = chunk["content"]
                                elif chunk["type"] == "token":
                                    response_text += chunk["content"]
                                    message_placeholder.markdown(response_text + "▌")
                                elif chunk["type"] == "error":
                                    st.error(chunk["content"])
                                    
                # Finalize output text
                if response_text:
                    message_placeholder.markdown(response_text)
                    
                    # Display retrieved references at bottom of message
                    if citations:
                        with st.expander("📚 View Grounded Reference Sources"):
                            for cit in citations:
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-header">
                                        📄 {cit['source_file']} &nbsp;|&nbsp; Chunk {cit['chunk_index']}
                                        <span class="source-score">L2 Distance: {cit['score']:.4f}</span>
                                    </div>
                                    <div class="source-body">
                                        {cit['text']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "citations": citations
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"Error communicating with backend: {str(e)}")
