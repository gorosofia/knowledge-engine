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
from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_dotenv()

# Configuración
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", 1024))
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", 0))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
N_RESULTS = int(os.getenv("N_RESULTS", 8))


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
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection("knowledge_base")
    except Exception as e:
        print(f"❌ Error cargando componentes: {e}")
        sys.exit(1)
    
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["documents"] or not results["documents"][0]:
        print("❌ No se encontró información relevante. Ejecuta build_index.py primero.")
        sys.exit(0)
    
    context_parts = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        source = metadata.get("source", "desconocido")
        context_parts.append(f"[Fuente: {source}]\n{doc}")
    
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
