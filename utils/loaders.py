from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader

def _read_pdf(p: Path) -> str:
    try:
        reader = PdfReader(str(p))
        return "\n".join([(pg.extract_text() or "") for pg in reader.pages])
    except Exception:
        return ""

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def load_documents(root: Path) -> List[Tuple[str, str]]:
    """Return list of (source_path, text) for files in root."""
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        sfx = p.suffix.lower()
        if sfx in [".txt", ".md", ".csv", ".log"]:
            print(f"Reading {p.name}...")
            out.append((str(p), _read_text(p)))
        elif sfx in [".pdf"]:
            txt = _read_pdf(p)
            print(f"Reading {p.name}... ({len(txt)} chars)")
            out.append((str(p), txt))
    return [(src, txt.strip()) for src, txt in out if txt and txt.strip()]
