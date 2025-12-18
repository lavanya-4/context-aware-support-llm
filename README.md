# 🧠 Context-Aware Support System (Hybrid Hierarchical RAG)

https://github.com/user-attachments/assets/5d839612-361c-4356-87e6-c543ebfbbfc4

A privacy-preserving, cost-efficient customer support assistant that combines **Edge AI** with **Cloud LLMs**.

This project implements a **Hierarchical Retrieval-Augmented Generation (RAG)** pipeline. Instead of sending raw retrieval data directly to expensive cloud providers, we use a locally running Small Language Model (SLM) to **read, refine, and compress** the context before final generation.

---

## 🚀 Key Highlights

*   **⚡️ 90% Cost Reduction**: By using a local SLM to filter irrelevant text, we reduce the token payload sent to paid APIs (e.g., GPT-4, Gemini) by **~89%**.
*   **🔒 Enhanced Privacy**: Sensitive document chunks are processed locally. Only a sanitized, relevant summary leaves your machine.
*   **🧠 Hybrid Architecture**:
    *   **Local (Edge):** Uses **MLX** to run Quantized Llama models on Apple Silicon for free, low-latency context refinement.
    *   **Cloud:** Uses **Google Gemini / Meta Llama 3** (via OpenRouter) for high-quality final answer generation.
*   **📚 Domain Specific**: Indexed on **SJSU Policy Documents** (Student Conduct Code, Academic Integrity) to answer complex institutional queries.

---

## 🛠️ Architecture

The system operates in a 4-stage pipeline:

1.  **Ingestion**: PDF documents are parsed and split into semantic chunks (500 tokens).
2.  **Retrieval**: User queries trigger a vector search (FAISS) to find the Top-5 most relevant chunks.
3.  **Refinement (The "Secret Sauce")**: 
    *   A local **Llama-3-Instruct** (or TinyLlama) reads the messy Top-5 chunks.
    *   It extracts *only* the key facts relevant to the query.
    *   This compresses ~2,500 tokens of noise into a ~250 token signal.
4.  **Generation**: The clean summary is sent to the Cloud LLM to generate a polite, customer-facing response with citations.

---

## 📊 Evaluation Results

We evaluated the system on a test set of 10 realistic student queries (e.g., "What are the rules on hazing?", "Can I appeal a grade?").

| Metric | Standard RAG | Hybrid RAG (Ours) | Impact |
| :--- | :--- | :--- | :--- |
| **Token Usage (per query)** | ~2,304 tokens | ~248 tokens | **10x Efficiency** |
| **Token Reduction** | -- | **89.2%** | **Massive Cost Savings** |
| **Recall@5** | -- | 60.0% | High Retrieval Accuracy |
| **Est. Cost (per 1k queries)** | ~$11.52 | **~$1.24** | **~90% Cheaper** |

*Note: Cost estimated based on GPT-4o input pricing.*

---

## ⚙️ Setup & Installation

### Prerequisites
*   Python 3.9+
*   Mac with Apple Silicon (recommended for MLX acceleration)
*   OpenRouter API Key

### 1. Clone & Install
```bash
git clone https://github.com/your-repo/context-aware-support-llm.git
cd context-aware-support-llm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_api_key_here
# Optional: Select embedding model
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🚦 Usage

### 1. Build the Knowledge Base
Ingest the PDFs from the `pdfs/` directory and build the FAISS index.
```bash
python indexer.py
```

### 2. Start the Server
Run the FastAPI backend (and load the local LLM).
```bash
python rag_server.py
```
*The server will start at `http://localhost:8000`*

### 3. Chat Interface
Open `http://localhost:8000` in your browser to interact with the support assistant.

### 4. Run Evaluation
To reproduce our metrics (Token Reduction & Recall):
```bash
python evaluate.py
```

---

## 📂 Project Structure

```
├── pdfs/                 # Source documents (SJSU Policies)
├── .rag_index/           # Generated FAISS vector index & metadata
├── static/               # Frontend HTML/CSS/JS
├── utils/
│   ├── loaders.py        # PDF parsing and text loading
│   └── text.py           # Token-aware chunking logic
├── indexer.py            # Script to build the vector database
├── rag_server.py         # Main API server & Orchestrator
├── evaluate.py           # Automated testing & metrics script
└── requirements.txt      # Python dependencies
```

## 👩‍💻 Technologies

*   **Backend:** FastAPI, Uvicorn
*   **Local Inference:** MLX (Machine Learning on X)
*   **Vector Search:** FAISS (Facebook AI Similarity Search)
*   **Embeddings:** Sentence-Transformers (HuggingFace)
*   **PDF Processing:** PyPDF
