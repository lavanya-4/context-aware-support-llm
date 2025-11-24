import os, json
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(".rag_index")
META_PATH = INDEX_DIR / "meta.json"
INDEX_PATH = INDEX_DIR / "faiss.index"
DIM_PATH = INDEX_DIR / "dim.txt"

TOP_K = 6
MAX_CTX = 6

class AskReq(BaseModel):
    query: str

class AskResp(BaseModel):
    answer: str
    citations: List[Dict]

app = FastAPI(title="Company RAG")

def load_index():
    if not (INDEX_PATH.exists() and META_PATH.exists() and DIM_PATH.exists()):
        raise RuntimeError("Index not found. Run: python indexer.py")
    index = faiss.read_index(str(INDEX_PATH))
    metas = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, metas

def slm_generate(prompt: str) -> str:
    base = os.getenv("SLM_BASE_URL", "http://localhost:9000")
    model = os.getenv("SLM_MODEL", "company-slm-v1")
    token = os.getenv("AUTH_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    r = requests.post(f"{base}/v1/chat/completions", json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

SYSTEM = """You are a precise assistant. Answer ONLY using the provided context.
If the answer is not in the context, say you don't know and suggest where to look internally.
Cite sources like [1], [2] at the end of supporting sentences.
"""

def build_prompt(query: str, hits: List[Tuple[float, Dict]]):
    hits = sorted(hits, key=lambda x: x[0], reverse=True)[:MAX_CTX]
    blocks, refs = [], []
    for i, (score, m) in enumerate(hits, 1):
        chunk = m["chunk"]
        source = m["source"]
        preview = " ".join(chunk.split()[:12])
        blocks.append(f"[{i}] {chunk}")
        refs.append({"n": i, "source": source, "preview": preview + " ..."})
    ctx = "\n\n".join(blocks)
    prompt = f"""{SYSTEM}

Query:
{query}

Context:
{ctx}

Instructions:
- Keep answers concise.
- Use inline citations like [1], [3] after sentences they support.
- If multiple chunks support a sentence, include multiple citations.

Answer now."""
    return prompt, refs

@app.on_event("startup")
def _startup():
    load_dotenv()
    global _index, _metas, _embedder
    _index, _metas = load_index()
    _embedder = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

def retrieve(query: str, top_k=TOP_K):
    qv = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = _index.search(qv, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append((float(score), _metas[idx]))
    return results

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    hits = retrieve(req.query)
    if not hits:
        return AskResp(answer="I couldn't find anything relevant in the index. Please add more documents.", citations=[])
    prompt, refs = build_prompt(req.query, hits)
    answer = slm_generate(prompt)
    cits = [{"source": r["source"], "preview": r["preview"]} for r in refs]
    return AskResp(answer=answer, citations=cits)
