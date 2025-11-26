import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from utils.loaders import load_documents
from utils.text import clean_text, chunk_by_tokens

DOC_DIRS = [Path("pdfs")]  # Scan both folders
INDEX_DIR = Path(".rag_index")
META_PATH = INDEX_DIR / "meta.json"
INDEX_PATH = INDEX_DIR / "faiss.index"
DIM_PATH = INDEX_DIR / "dim.txt"

CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50

def build():
    load_dotenv()
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(embed_model_name)

    all_docs = []
    for d in DOC_DIRS:
        if d.exists():
            print(f"Loading documents from {d}...")
            all_docs.extend(load_documents(d))
        else:
            print(f"[!] {d}/ not found. Skipping.")

    if not all_docs:
        print(f"[!] No documents found in {DOC_DIRS}")
        return

    chunks: List[str] = []
    metas: List[Dict] = []

    for src, txt in all_docs:
        txt = clean_text(txt)
        for ch in chunk_by_tokens(txt, CHUNK_TOKENS, CHUNK_OVERLAP):
            chunks.append(ch)
            metas.append({"source": src, "chunk": ch})

    print(f"[i] Total chunks: {len(chunks)}")
    X = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    dim = X.shape[1]

    INDEX_DIR.mkdir(exist_ok=True)
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    DIM_PATH.write_text(str(dim), encoding="utf-8")
    print("[✓] Index built.")

if __name__ == "__main__":
    build()
