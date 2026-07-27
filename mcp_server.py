#!/usr/bin/env python3
"""
Servidor MCP para Knowledge Engine.
Expone 'search_knowledge' para que agentes externos consulten tu base de conocimiento.

Transporte: stdio
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("knowledge-engine")

load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import chromadb
from openai import OpenAI

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", 1024))
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", 0))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
N_RESULTS = int(os.getenv("N_RESULTS", 8))

logger.info("Iniciando Knowledge Engine MCP Server...")

try:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = chroma_client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    ai_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    logger.info("Componentes inicializados correctamente.")
except Exception as e:
    logger.error(f"Error durante la inicialización: {e}")
    sys.exit(1)

app = Server("knowledge-engine")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge",
            description=(
                "Busca información en la base de conocimiento personal (notas de Obsidian) "
                "y devuelve una respuesta generada con DeepSeek. "
                "Úsala cuando necesites responder preguntas sobre tus notas o documentos."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pregunta o términos de búsqueda"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Número máximo de fragmentos a recuperar (predeterminado: 8)",
                        "default": 8
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "search_knowledge":
        return [TextContent(
            type="text",
            text=f"Error: Herramienta desconocida '{name}'. Disponible: search_knowledge"
        )]

    query = arguments.get("query", "")
    n_results = arguments.get("n_results", 8)

    if not query:
        return [TextContent(type="text", text="Error: El parámetro 'query' es requerido.")]

    try:
        logger.info(f"Consulta recibida: {query}")

        query_embedding_response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": query},
            timeout=120,
        )
        query_embedding_response.raise_for_status()
        query_embedding = [query_embedding_response.json()["embedding"]]

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, N_RESULTS),
            include=["documents", "metadatas"]
        )

        if not results["documents"] or not results["documents"][0]:
            return [TextContent(
                type="text",
                text="No se encontró información relevante en la base de conocimiento."
            )]

        context_parts = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            source = metadata.get("source", "desconocido")
            context_parts.append(f"[Fuente: {source}]\n{doc}")

        context = "\n\n---\n\n".join(context_parts)

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
        logger.info(f"Respuesta generada ({len(answer)} caracteres)")
        return [TextContent(type="text", text=answer)]

    except Exception as e:
        logger.error(f"Error en search_knowledge: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error interno: {str(e)}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
