import os
import sys
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path so we can import modules
sys.path.append(str(Path(__file__).parent))

# Import specific components to test
from utils.loaders import load_documents
from utils.text import chunk_by_tokens
from rag_server import load_index
from sentence_transformers import SentenceTransformer
import faiss

# ANSI Colors for pretty printing
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def print_status(component, status, msg=""):
    color = GREEN if status == "PASS" else RED
    print(f"[{color}{status}{RESET}] {component}: {msg}")

async def test_1_ingestion_parsing():
    print(f"\n{YELLOW}--- Testing Module 1: Ingestion & Parsing ---{RESET}")
    pdf_dir = Path("pdfs")
    
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        print_status("Ingestion", "FAIL", "No PDFs found in 'pdfs/' directory")
        return False

    try:
        t0 = time.time()
        docs = load_documents(pdf_dir)
        if not docs:
            print_status("Ingestion", "FAIL", "load_documents returned empty list")
            return False
            
        print(f"   Loaded {len(docs)} documents.")
        sample_src, sample_text = docs[0]
        print(f"   Sample Source: {sample_src}")
        print(f"   Sample Text Length: {len(sample_text)} chars")
        
        # Test Chunking
        chunks = chunk_by_tokens(sample_text, chunk_tokens=100, overlap=10)
        print(f"   Chunking Test: {len(chunks)} chunks created from sample.")
        
        print_status("Ingestion", "PASS", f"Parsed {len(docs)} docs in {time.time()-t0:.2f}s")
        return True
    except Exception as e:
        print_status("Ingestion", "FAIL", str(e))
        return False

async def test_2_retrieval_index():
    print(f"\n{YELLOW}--- Testing Module 2: Retrieval & Indexing ---{RESET}")
    try:
        t0 = time.time()
        # 1. Check index files
        index_path = Path(".rag_index/faiss.index")
        if not index_path.exists():
             print_status("Retrieval", "FAIL", "Index file not found. Run 'python indexer.py' first.")
             return False

        # 2. Load Index
        index, metas = load_index()
        print(f"   Index Loaded. Size: {index.ntotal} vectors.")
        
        # 3. Load Embedder
        print("   Loading Embedding Model (MiniLM-L6-v2)...")
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # 4. Perform Query
        query = "hearing board"
        print(f"   Testing Query: '{query}'")
        qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, 3) 
        
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx != -1:
                hits.append(metas[idx])
        
        if hits:
            print(f"   Top Result Source: {hits[0]['source']}")
            print(f"   Top Result Preview: {hits[0]['chunk'][:100]}...")
            print_status("Retrieval", "PASS", f"Retrieved {len(hits)} hits in {time.time()-t0:.2f}s")
            return True
        else:
            print_status("Retrieval", "FAIL", "No results found for test query")
            return False
            
    except Exception as e:
        print_status("Retrieval", "FAIL", str(e))
        return False

async def test_3_slm_model_loading():
    print(f"\n{YELLOW}--- Testing Module 3: Local SLM Loading (MLX) ---{RESET}")
    print("   Note: This loads the actual MLX model into memory. It may take a few seconds.")
    try:
        from mlx_lm import load, generate
        model_path = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
        
        t0 = time.time()
        print(f"   Loading {model_path}...")
        model, tokenizer = load(model_path)
        print_status("SLM Loading", "PASS", f"Model loaded in {time.time()-t0:.2f}s")
        
        # Quick generation test
        prompt = "Hello, are you working?"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
        print(f"   Test Output: {response.strip()}")
        return True
    except ImportError:
         print_status("SLM Loading", "FAIL", "mlx_lm not installed or not supported.")
         return False
    except Exception as e:
        print_status("SLM Loading", "FAIL", str(e))
        return False

async def test_4_external_api():
    print(f"\n{YELLOW}--- Testing Module 4: External LLM Connection ---{RESET}")
    # We can reuse the function from rag_server if we mock the env
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print_status("External LLM", "SKIP", "OPENROUTER_API_KEY not found in .env")
        return True

    from rag_server import external_llm_generate
    
    try:
        summary = "This is a test summary. The user asked for a status check."
        query = "Status check"
        print("   Sending test request to OpenRouter...")
        t0 = time.time()
        ans = await external_llm_generate(query, summary)
        print(f"   Response Preview: {ans[:100]}...")
        print_status("External LLM", "PASS", f"Response received in {time.time()-t0:.2f}s")
        return True
    except Exception as e:
        print_status("External LLM", "FAIL", str(e))
        return False

async def main():
    print(f"{YELLOW}Starting Full Component Test...{RESET}")
    
    # 1. Ingestion
    if not await test_1_ingestion_parsing():
        print("Stopping tests due to Ingestion failure.")
        return

    # 2. Retrieval
    if not await test_2_retrieval_index():
        print("Stopping tests due to Retrieval failure.")
        return

    # 3. SLM
    await test_3_slm_model_loading()

    # 4. External API
    await test_4_external_api()
    
    print(f"\n{GREEN}All Tests Completed.{RESET}")

if __name__ == "__main__":
    asyncio.run(main())
