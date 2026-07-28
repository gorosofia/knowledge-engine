#!/usr/bin/env python3
"""
Indexa documentos (txt, md, pdf) en ChromaDB usando embeddings locales de Ollama.

Indexación incremental: solo procesa archivos nuevos o modificados.
Los embeddings se generan en paralelo para acelerar el proceso.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import chromadb
import requests
from pypdf import PdfReader

load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "../knowledge-vault"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_WORKERS = int(os.getenv("INDEX_MAX_WORKERS", 4))

MANIFEST_FILE = "index_manifest.json"


def get_ollama_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("embedding", [])


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


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print("⚠️  Manifest corrupto, empezando de cero.")
    return {}


def save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def process_file(file_path: Path, docs_dir: Path) -> tuple[list[str], list[dict], list[str]]:
    """Chunkea un archivo y devuelve (documents, metadatas, ids)."""
    text = extract_text(file_path)
    if not text.strip():
        return [], [], []

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    rel_path = file_path.relative_to(docs_dir)
    file_hash = get_file_hash(file_path)

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{rel_path}_{i}_{file_hash}"
        documents.append(chunk)
        metadatas.append({
            "source": str(rel_path),
            "chunk_index": i,
            "total_chunks": len(chunks),
        })
        ids.append(chunk_id)

    return documents, metadatas, ids


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
        metadata={"hnsw:space": "cosine"},
    )

    manifest_path = CHROMA_DB_PATH / MANIFEST_FILE
    manifest = load_manifest(manifest_path)

    supported = {".txt", ".md", ".pdf"}
    current_files = sorted([
        f for f in DOCS_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in supported
    ])

    if not current_files:
        print("❌ No se encontraron documentos en la carpeta indicada.")
        sys.exit(1)

    print(f"📄 Archivos en vault: {len(current_files)}")
    print(f"📋 Archivos en manifest: {len(manifest)}")

    # ── 1. Identificar cambios ─────────────────────────────────────────

    current_map: dict[str, Path] = {}
    for f in current_files:
        rel = str(f.relative_to(DOCS_DIR))
        current_map[rel] = f

    to_process: list[Path] = []
    to_delete_sources: list[str] = []
    skipped = 0

    for rel_path, file_path in sorted(current_map.items()):
        if rel_path not in manifest:
            to_process.append(file_path)
        else:
            current_hash = get_file_hash(file_path)
            if current_hash != manifest[rel_path]["hash"]:
                to_process.append(file_path)
                to_delete_sources.append(rel_path)
            else:
                skipped += 1

    for rel_path in manifest:
        if rel_path not in current_map:
            to_delete_sources.append(rel_path)

    # ── 2. Limpiar fragmentos huérfanos ────────────────────────────────

    deleted_total = 0
    for source in to_delete_sources:
        try:
            old_docs = collection.get(where={"source": source})
            if old_docs and old_docs["ids"]:
                collection.delete(ids=old_docs["ids"])
                deleted_total += len(old_docs["ids"])
                print(f"   🗑️  {source}: {len(old_docs['ids'])} fragmento(s) eliminados")
        except Exception as e:
            print(f"   ⚠️  Error al limpiar {source}: {e}")

    if deleted_total > 0:
        print(f"🗑️  Total eliminado: {deleted_total} fragmento(s) de archivos obsoletos")

    if not to_process:
        print(f"⏭️  Sin cambios desde el último index ({skipped} archivos sin modificar).")
        total = collection.count()
        print(f"✅ Indexación completada. Fragmentos en la base de datos: {total}")
        return

    # ── 3. Procesar archivos nuevos/modificados ─────────────────────────

    print(f"🔍 Procesando {len(to_process)} archivo(s) nuevo(s) o modificado(s)...")

    all_documents = []
    all_metadatas = []
    all_ids = []
    file_records: dict[str, dict] = {}

    for file_path in to_process:
        rel_path = str(file_path.relative_to(DOCS_DIR))
        docs, metas, ids = process_file(file_path, DOCS_DIR)
        if docs:
            all_documents.extend(docs)
            all_metadatas.extend(metas)
            all_ids.extend(ids)
            file_records[rel_path] = {
                "hash": get_file_hash(file_path),
                "chunks": len(docs),
            }
            print(f"   📄 {rel_path}: {len(docs)} fragmento(s)")
        else:
            print(f"   ⚠️  {rel_path}: sin contenido extraíble, saltando")

    if not all_documents:
        print("❌ No se extrajo texto de ningún archivo.")
        sys.exit(1)

    print(f"🔍 Generando embeddings para {len(all_documents)} fragmentos (Ollama local, {MAX_WORKERS} workers)...")
    embeddings_list = [None] * len(all_documents)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(get_ollama_embedding, doc): i
            for i, doc in enumerate(all_documents)
        }
        done = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                embeddings_list[idx] = future.result()
            except Exception as e:
                print(f"\n❌ Error generando embedding para fragmento {idx}: {e}")
                sys.exit(1)
            done += 1
            print(f"\r   - Fragmento {done}/{len(all_documents)}", end="", flush=True)

    print()

    print("💾 Almacenando en ChromaDB...")
    collection.add(
        documents=all_documents,
        embeddings=embeddings_list,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    # ── 4. Actualizar manifest ──────────────────────────────────────────

    for rel_path in to_delete_sources:
        manifest.pop(rel_path, None)
    manifest.update(file_records)
    save_manifest(manifest_path, manifest)

    total = collection.count()
    print(f"✅ Indexación completada. Fragmentos en la base de datos: {total}")


if __name__ == "__main__":
    main()
