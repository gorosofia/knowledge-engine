#!/usr/bin/env python3
"""
Consulta la base de conocimiento y genera una respuesta con DeepSeek.
Optimizado para coste: usa deepseek-v4-flash, max_tokens limitado y temperatura 0.

Uso:
    python ask.py "¿Qué es la inteligencia artificial?"
    echo "¿Qué es la IA?" | python ask.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
import requests
from openai import OpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", 1024))
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", 0))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
N_RESULTS = int(os.getenv("N_RESULTS", 24))


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        try:
            query = input("❓ Introduce tu pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Entrada cancelada.")
            sys.exit(1)
    
    if not query:
        print("❌ Debes proporcionar una pregunta.")
        sys.exit(1)
    
    if not DEEPSEEK_API_KEY:
        print("❌ Error: DEEPSEEK_API_KEY no está configurada en el archivo .env")
        sys.exit(1)
    
    print("🔍 Buscando información relevante...")
    try:
        ai_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection("knowledge_base")
    except Exception as e:
        print(f"❌ Error cargando componentes: {e}")
        sys.exit(1)
    
    try:
        query_embedding_response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": query},
            timeout=120,
        )
        query_embedding_response.raise_for_status()
        query_embedding = [query_embedding_response.json()["embedding"]]
    except Exception as e:
        print(f"❌ Error generando embedding de la consulta: {e}")
        sys.exit(1)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=N_RESULTS * 2,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["documents"] or not results["documents"][0]:
        print("❌ No se encontró información relevante. Ejecuta build_index.py primero.")
        sys.exit(0)
    
    seen_sources: set[str] = set()
    context_parts = []
    deduped_count = 0
    
    for i, doc in enumerate(results["documents"][0]):
        if len(context_parts) >= N_RESULTS:
            break
        metadata = results["metadatas"][0][i]
        source = metadata.get("source", "desconocido")
        if source not in seen_sources:
            seen_sources.add(source)
            context_parts.append(f"[Fuente: {source}]\n{doc}")
        else:
            deduped_count += 1
    
    for i, doc in enumerate(results["documents"][0]):
        if len(context_parts) >= N_RESULTS * 2:
            break
        metadata = results["metadatas"][0][i]
        source = metadata.get("source", "desconocido")
        if source in seen_sources:
            already_included = any(source in p for p in context_parts if p.startswith(f"[Fuente: {source}]"))
            if not already_included:
                context_parts.append(f"[Fuente: {source}]\n{doc}")
    
    print(f"   ({len(seen_sources)} fuentes únicas, {deduped_count} duplicados omitidos)")
    context = "\n\n---\n\n".join(context_parts)
    
    print("🤖 Generando respuesta con DeepSeek...")
    try:
        ai_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        
        response = ai_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=DEEPSEEK_MAX_TOKENS,
            temperature=DEEPSEEK_TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Asistente de investigación. Responde SOLO con el contexto proporcionado. "
                        "Si no sabes la respuesta, dilo. Cita fuentes entre [corchetes]."
                    )
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPregunta: {query}"
                }
            ]
        )
        
        answer = response.choices[0].message.content
        print("\n" + "=" * 70)
        print(answer)
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"❌ Error al llamar a DeepSeek: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
