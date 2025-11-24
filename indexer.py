import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from utils.loaders import load_documents
from utils.text import clean_text, chunk_by_tokens

DOC_DIR = Path("docs")                   # <-- must be at project root
INDEX_DIR = Path(".rag_index")
META_PATH = INDEX_DIR / "meta.json"
INDEX_PATH = INDEX_DIR / "faiss.index"
DIM_PATH = INDEX_DIR / "dim.txt"

CHUNK_TOKENS = 400
CHUNK_OVERLAP = 40

def build():
    load_dotenv()
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(embed_model_name)

    if not DOC_DIR.exists():
        print(f"[!] {DOC_DIR}/ not found. Create it and add some files.")
        return

    docs = load_documents(DOC_DIR)
    if not docs:
        print(f"[!] No documents found in {DOC_DIR}/")
        return

    chunks: List[str] = []
    metas: List[Dict] = []

    for src, txt in docs:
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
