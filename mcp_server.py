#!/usr/bin/env python3
"""
Servidor MCP (Model Context Protocol) para Knowledge Engine.
Expone la herramienta 'search_knowledge' para que agentes externos 
(Claude Code, Cursor, etc.) consulten tu base de conocimiento.

Transporte: stdio (estándar para clientes MCP locales)
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import asyncio

# Logging a stderr para no ensuciar el stdout (protocolo MCP)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("knowledge-engine")

load_dotenv()

# Importaciones de terceros
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
N_RESULTS = int(os.getenv("N_RESULTS", 5))

# ──────────────────────────────────────────────
# Inicialización de componentes (una sola vez)
# ──────────────────────────────────────────────
logger.info("Iniciando Knowledge Engine MCP Server...")

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
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

# ──────────────────────────────────────────────
# Definición del servidor MCP
# ──────────────────────────────────────────────
app = Server("knowledge-engine")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Lista las herramientas que ofrece este servidor."""
    return [
        Tool(
            name="search_knowledge",
            description=(
                "Busca información en la base de conocimiento personal (notas de Obsidian indexadas) "
                "y devuelve una respuesta generada con DeepSeek. "
                "Usa esta herramienta cuando necesites responder preguntas sobre tus notas, "
                "documentos o cualquier conocimiento que hayas indexado previamente."
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
                        "description": "Número máximo de fragmentos a recuperar (predeterminado: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Ejecuta la herramienta solicitada."""
    if name != "search_knowledge":
        return [TextContent(
            type="text",
            text=f"Error: Herramienta desconocida '{name}'. Disponible: search_knowledge"
        )]
    
    query = arguments.get("query", "")
    n_results = arguments.get("n_results", 5)
    
    if not query:
        return [TextContent(type="text", text="Error: El parámetro 'query' es requerido.")]
    
    try:
        logger.info(f"Consulta recibida: {query}")
        
        # 1. Generar embedding de la consulta
        query_embedding = embedding_model.encode([query]).tolist()
        
        # 2. Buscar en ChromaDB
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
        
        # 3. Construir contexto
        context_parts = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            source = metadata.get("source", "desconocido")
            context_parts.append(f"[Fuente: {source}]\n{doc}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # 4. Generar respuesta con DeepSeek
        response = ai_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente de investigación. Responde basándote ÚNICAMENTE en el contexto "
                        "proporcionado. Si no sabes la respuesta, dilo claramente. Cita las fuentes."
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
        logger.info(f"Respuesta generada ({len(answer)} caracteres)")
        return [TextContent(type="text", text=answer)]
    
    except Exception as e:
        logger.error(f"Error en search_knowledge: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error interno: {str(e)}")]


async def main():
    """Punto de entrada: ejecuta el servidor MCP por stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
