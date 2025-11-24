import re
from typing import List
import tiktoken

def clean_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_by_tokens(text: str, chunk_tokens=400, overlap=40, enc_name="cl100k_base") -> List[str]:
    enc = tiktoken.get_encoding(enc_name)
    ids = enc.encode(text)
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + chunk_tokens, len(ids))
        chunk = enc.decode(ids[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(ids):
            break
        start = end - overlap
    return chunks
