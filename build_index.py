#!/usr/bin/env python3
"""
Indexa documentos (txt, md, pdf) en ChromaDB usando embeddings locales de BAAI/bge-m3.
Los embeddings se generan en tu máquina, por lo que no hay coste de API.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import requests
from pypdf import PdfReader
import hashlib

load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "../knowledge-vault"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


def get_ollama_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("embedding", [])


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    try:
        if suffix in (".txt", ".md"):
            return file_path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            print(f"⚠️  Formato no soportado: {file_path.name} ({suffix})")
            return ""
    except Exception as e:
        print(f"❌ Error leyendo {file_path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def get_file_hash(file_path: Path) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def main():
    print(f"📂 Carpeta de documentos: {DOCS_DIR.resolve()}")
    
    if not DOCS_DIR.exists():
        print(f"❌ Error: La carpeta {DOCS_DIR} no existe.")
        sys.exit(1)
    
    print(f"🧠 Usando embeddings locales por Ollama: {OLLAMA_EMBEDDING_MODEL}")
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        print(f"❌ No se pudo conectar a Ollama ({OLLAMA_URL}): {e}")
        sys.exit(1)
    
    print(f"💾 Inicializando base de datos vectorial en: {CHROMA_DB_PATH.resolve()}")
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    supported = {".txt", ".md", ".pdf"}
    files = sorted([
        f for f in DOCS_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in supported
    ])
    
    if not files:
        print("❌ No se encontraron documentos en la carpeta indicada.")
        sys.exit(1)
    
    print(f"📄 Archivos encontrados: {len(files)}")
    
    documents = []
    metadatas = []
    ids = []
    
    for file_path in files:
        text = extract_text(file_path)
        if not text.strip():
            continue
        
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        rel_path = file_path.relative_to(DOCS_DIR)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}_{i}_{get_file_hash(file_path)}"
            documents.append(chunk)
            metadatas.append({
                "source": str(rel_path),
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            ids.append(chunk_id)
    
    if not documents:
        print("❌ No se extrajo texto de ningún archivo.")
        sys.exit(1)
    
    print(f"🔍 Generando embeddings para {len(documents)} fragmentos (Ollama local)...")
    embeddings = []
    for i, document in enumerate(documents, 1):
        print(f"   - Fragmento {i}/{len(documents)}")
        embeddings.append(get_ollama_embedding(document))
    
    print("💾 Almacenando en ChromaDB...")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    total = collection.count()
    print(f"✅ Indexación completada. Fragmentos en la base de datos: {total}")


if __name__ == "__main__":
    main()
