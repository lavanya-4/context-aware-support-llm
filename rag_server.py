import os, json, re
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.staticfiles import StaticFiles

# MLX imports
from mlx_lm import load, generate

# Configurations
INDEX_DIR = Path(".rag_index")
META_PATH = INDEX_DIR / "meta.json"
INDEX_PATH = INDEX_DIR / "faiss.index"
DIM_PATH = INDEX_DIR / "dim.txt"
# Use an MLX-compatible model
BASE_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
ADAPTER_PATH = "./adaptor" 

TOP_K = 6
MAX_CTX = 6

SYSTEM = """You are a precise assistant. Answer ONLY using the provided context.
If the answer is not in the context, say you don't know and suggest where to look internally.
Cite sources like [1], [2] at the end of supporting sentences.
"""

class AskReq(BaseModel):
    message: str

class AskResp(BaseModel):
    reply: str
    citations: List[Dict]

# Globals for loaded models
_llama_tokenizer = None
_llama_model = None
_index = None
_metas = None
_embedder = None

def load_index():
    if not (INDEX_PATH.exists() and META_PATH.exists() and DIM_PATH.exists()):
        raise RuntimeError("Index not found. Run: python indexer.py")
    index = faiss.read_index(str(INDEX_PATH))
    metas = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, metas

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    global _index, _metas, _embedder, _llama_tokenizer, _llama_model
    _index, _metas = load_index()
    _embedder = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    
    print(f"Loading MLX model: {BASE_MODEL}...")
    # Try to load with adapter if it exists and is compatible, otherwise load base
    # Note: MLX adapter loading might require specific format. 
    # For now, we load the base model. To use the adapter, it needs to be converted to MLX format.
    _llama_model, _llama_tokenizer = load(BASE_MODEL)
    print("Model loaded.")
    
    yield
    # Clean up resources if needed

app = FastAPI(title="Company RAG", lifespan=lifespan)

def slm_generate(prompt: str, max_new_tokens=256) -> str:
    global _llama_tokenizer, _llama_model
    
    # mlx_lm.generate takes the model and tokenizer directly
    completion = generate(
        _llama_model,
        _llama_tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        verbose=False
    )
    
    # Optional: strip "assistant" header if used in your LoRA prompt style
    # Optional: strip "assistant" header if used in your LoRA prompt style
    if "<|start_header_id|>assistant<|end_header_id|>" in completion:
        completion = completion.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    
    # Aggressively clean up trailing backticks and whitespace
    completion = re.sub(r'[`\s]+$', '', completion)
    return completion

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

def retrieve(query: str, top_k=TOP_K):
    qv = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = _index.search(qv, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append((float(score), _metas[idx]))
    return results

@app.post("/api/ask", response_model=AskResp)
def ask(req: AskReq):
    hits = retrieve(req.message)
    if not hits:
        return AskResp(reply="I couldn't find anything relevant in the index. Please add more documents.", citations=[])
    prompt, refs = build_prompt(req.message, hits)
    answer = slm_generate(prompt)
    cits = [{"source": r["source"], "preview": r["preview"]} for r in refs]
    return AskResp(reply=answer, citations=cits)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)