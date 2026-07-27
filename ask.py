#!/usr/bin/env python3
"""
Consulta la base de conocimiento y genera una respuesta con DeepSeek.
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
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
N_RESULTS = int(os.getenv("N_RESULTS", 5))


def main():
    # Obtener pregunta desde argumentos o stdin
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
    
    # Cargar embedding y base de datos
    print("🔍 Buscando información relevante...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection("knowledge_base")
    except Exception as e:
        print(f"❌ Error cargando componentes: {e}")
        sys.exit(1)
    
    # Buscar fragmentos similares
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["documents"] or not results["documents"][0]:
        print("❌ No se encontró información relevante. Asegúrate de haber ejecutado build_index.py primero.")
        sys.exit(0)
    
    # Construir contexto
    context_parts = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        source = metadata.get("source", "desconocido")
        context_parts.append(f"[Fuente: {source}]\n{doc}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Llamar a DeepSeek
    print("🤖 Generando respuesta con DeepSeek...")
    try:
        ai_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        
        response = ai_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente de investigación. Responde basándote ÚNICAMENTE en el contexto "
                        "proporcionado. Si la información no está disponible, indícalo claramente. "
                        "Cita las fuentes cuando sea posible."
                    )
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPregunta: {query}"
                }
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        # Mostrar resultado
        print("\n" + "=" * 70)
        print(answer)
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"❌ Error al llamar a DeepSeek: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
