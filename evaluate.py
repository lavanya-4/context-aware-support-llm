import asyncio
import json
import time
import os
import faiss
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

INDEX_DIR = Path(".rag_index")
META_PATH = INDEX_DIR / "meta.json"
INDEX_PATH = INDEX_DIR / "faiss.index"
DATASET_PATH = Path("test_dataset.json")

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def load_resources():
    print(">>> Loading Resources...")
    if not INDEX_PATH.exists():
        raise FileNotFoundError("Index not found. Run indexer.py first.")
    index = faiss.read_index(str(INDEX_PATH))
    metas = json.loads(META_PATH.read_text(encoding="utf-8"))
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print(">>> Loading Local SLM (MLX)...")
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    
    return index, metas, embedder, model, tokenizer

def calculate_recall(retrieved_chunks: List[str], ground_truth_keywords: List[str]) -> bool:
    for chunk in retrieved_chunks:
        chunk_lower = chunk.lower()
        matches = sum(1 for kw in ground_truth_keywords if kw.lower() in chunk_lower)
        if matches >= (len(ground_truth_keywords) / 2):
            return True
    return False

async def evaluate():
    try:
        index, metas, embedder, slm_model, slm_tokenizer = load_resources()
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    if not DATASET_PATH.exists():
        print("Dataset not found.")
        return
    dataset = json.loads(DATASET_PATH.read_text())
    
    results = []
    
    print("\n>>> Starting Evaluation on", len(dataset), "queries...")
    print(f"{'ID':<4} | {'Query':<40} | {'Recall':<6} | {'Tokens (RAG)':<12} | {'Tokens (Hybrid)':<15} | {'Reduction':<10}")
    print("-" * 110)

    total_rag_tokens = 0
    total_hybrid_tokens = 0
    total_recall_hits = 0
    
    for item in dataset:
        q_id = item["id"]
        query = item["query"]
        keywords = item["ground_truth_keywords"]
        
        qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, 5)
        
        retrieved_hits = []
        retrieved_text_blobs = []
        
        for idx in I[0]:
            if idx != -1:
                chunk_text = metas[idx]["chunk"]
                retrieved_hits.append(metas[idx])
                retrieved_text_blobs.append(chunk_text)
        
        is_relevant = calculate_recall(retrieved_text_blobs, keywords)
        if is_relevant:
            total_recall_hits += 1
            
        raw_context = "\n\n".join(retrieved_text_blobs)
        tokens_rag = count_tokens(raw_context)
        
        prompt = f"""You are a helper. Extract only information relevant to: "{query}" from the text below.
        
        Text:
        {raw_context[:6000]} 
        
        Summary:"""
        
        refined_summary = generate(slm_model, slm_tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        tokens_hybrid = count_tokens(refined_summary)
        
        reduction_pct = ((tokens_rag - tokens_hybrid) / tokens_rag) * 100 if tokens_rag > 0 else 0
        
        total_rag_tokens += tokens_rag
        total_hybrid_tokens += tokens_hybrid
        
        print(f"{q_id:<4} | {query[:37]+'...':<40} | {str(is_relevant):<6} | {tokens_rag:<12} | {tokens_hybrid:<15} | {reduction_pct:.1f}%")
        
        results.append({
            "id": q_id,
            "recall": is_relevant,
            "tokens_rag": tokens_rag,
            "tokens_hybrid": tokens_hybrid,
            "reduction": reduction_pct
        })

    avg_reduction = (total_rag_tokens - total_hybrid_tokens) / total_rag_tokens * 100
    recall_at_5 = (total_recall_hits / len(dataset)) * 100
    
    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    print(f"Total Queries: {len(dataset)}")
    print(f"Recall@5: {recall_at_5:.1f}%")
    print("-" * 30)
    print(f"Total Tokens (Standard RAG): {total_rag_tokens}")
    print(f"Total Tokens (Hybrid RAG):   {total_hybrid_tokens}")
    print(f"Total Token Reduction:       {avg_reduction:.1f}%")
    print("="*50)
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(evaluate())
