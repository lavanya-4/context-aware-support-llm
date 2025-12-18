import os, json, re, time, asyncio
import httpx
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

TOP_K = 5
MAX_CTX = 5

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

def slm_refine_context(query: str, hits: List[Tuple[float, Dict]]) -> Tuple[str, List[Dict]]:
    """
    Uses the local SLM to read the retrieved chunks and generate a refined summary 
    containing only the key information relevant to the query.
    Returns the refined summary and the list of original hits (for citations).
    """
    hits = sorted(hits, key=lambda x: x[0], reverse=True)[:MAX_CTX]
    
    # Prepare context for refinement
    blocks = []
    for i, (score, m) in enumerate(hits, 1):
        chunk = m["chunk"]
        blocks.append(f"[{i}] {chunk}")
    ctx = "\n\n".join(blocks)
    
    prompt = f"""You are a helpful assistant.
Query: {query}

Contexts:
{ctx}

Task: Read the contexts above and extract ONLY the key information relevant to answering the query. 
Synthesize this into a concise summary. Do not include irrelevant details.
If the contexts don't contain the answer, say "No relevant information found."
Refined Summary:"""

    refined_summary = slm_generate(prompt, max_new_tokens=256)
    print(f"SLM Refined Summary: {refined_summary}")
    
    # We return all hits as potential citations, but the LLM sees the summary
    return refined_summary, [h[1] for h in hits]

async def external_llm_generate(query: str, context_summary: str) -> str:
    """
    Calls OpenRouter to generate the final answer using the refined summary from the SLM.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set."

    system_prompt = """You are a precise assistant. Answer the user's query using ONLY the provided Context Summary.
If the summary says "No relevant information found", state that you don't know."""

    user_message = f"""Query:
{query}

Context Summary:
{context_summary}

Answer:"""

    max_retries = 3
    base_delay = 2
    
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        # "HTTP-Referer": "http://localhost:8000", # Optional
                    },
                    json={
                        "model": "meta-llama/llama-3.2-3b-instruct:free", # Switched to Llama 3.2 3B (Free) to avoid rate limits
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.1
                    },
                    timeout=30.0
                )
                if resp.status_code == 429:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limited (429). Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        return "Error: Rate limit exceeded (429) after retries."
                        
                if resp.status_code != 200:
                    print(f"OpenRouter Error Status: {resp.status_code}")
                    print(f"OpenRouter Error Body: {resp.text}")
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"OpenRouter Error: {e}")
                if attempt < max_retries - 1:
                     await asyncio.sleep(1)
                     continue
                return f"Error contacting external LLM: {e}"

@app.post("/api/ask", response_model=AskResp)
async def ask(req: AskReq):
    start_total = time.time()
    
    # 1. Retrieve 5 chunks
    t0 = time.time()
    hits = retrieve(req.message, top_k=TOP_K)
    print(f"[Timing] Retrieval took: {time.time() - t0:.2f}s")
    
    if not hits:
        return AskResp(reply="I couldn't find anything relevant in the index.", citations=[])
    
    # 2. SLM refines context (Summarization/Extraction)
    t1 = time.time()
    refined_summary, all_metas = slm_refine_context(req.message, hits)
    print(f"[Timing] SLM Refinement took: {time.time() - t1:.2f}s")
    
    # 3. External LLM generates answer using the summary
    t2 = time.time()
    answer = await external_llm_generate(req.message, refined_summary)
    print(f"[Timing] External LLM took: {time.time() - t2:.2f}s")
    
    print(f"[Timing] Total Request took: {time.time() - start_total:.2f}s")
    
    # Prepare citations (we show the top 3 original sources as references)
    cits = [{"source": m["source"], "preview": " ".join(m["chunk"].split()[:12]) + " ..."} for m in all_metas[:3]]
    
    return AskResp(reply=answer, citations=cits)


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)