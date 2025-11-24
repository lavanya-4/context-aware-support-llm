# 🧠 context-aware-support-llm

This repository contains the implementation of **Customer Support Automation using LLMs and Intelligent Context Management**.  
It is a **Retrieval-Augmented Generation (RAG)** system that automates customer support by combining **Large Language Models (LLMs)** with **intelligent context management** to deliver context-aware answers using company-specific documents.

## 🚀 Features
- 💬 Natural-language question answering  
- 📄 Automatic document ingestion and chunking  
- 🔍 Semantic search using FAISS and SentenceTransformers  
- 🧠 Context-aware response generation with a Small Language Model (SLM)  
- ⚙️ RESTful API built using FastAPI

## ⚙️ Setup & Execution
```bash
pip install -r requirements.txt
python indexer.py
uvicorn rag_server:app --reload --port 8000

Technologies Used:

Python, FastAPI, Uvicorn
FAISS, SentenceTransformers
dotenv, pypdf, tiktoken

💡 Future Enhancements

Connect to the company’s real SLM endpoint

Add contextual memory and analytics dashboard

Integrate with customer chat or support platforms
