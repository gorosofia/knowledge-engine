#!/usr/bin/env python3
"""
Servidor MCP para Knowledge Engine + Memory + Tracker FA.
Expone herramientas de búsqueda de conocimiento, memoria de agentes y tracking de circuit breakers.
Transporte: stdio
"""

import os, sys, json, uuid, logging, asyncio
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
logger = logging.getLogger("knowledge-engine")

load_dotenv()

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool, TextContent, CallToolResult, CallToolRequestParams,
    ListToolsResult, PaginatedRequestParams
)
import chromadb
from openai import OpenAI

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", 1024))
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", 0))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
N_RESULTS = int(os.getenv("N_RESULTS", 24))
FA_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
TRACKER_DIR = Path(os.getenv("TRACKER_DIR", str(FA_ROOT / ".kilo" / "tracker")))

KNOWLEDGE_COLLECTION = "knowledge_base"
AGENT_MEMORY_COLLECTION = "agent_memory"

logger.info("Iniciando Knowledge Engine + Memory + Tracker MCP Server...")

try:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    kb_collection = chroma_client.get_or_create_collection(
        name=KNOWLEDGE_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    mem_collection = chroma_client.get_or_create_collection(
        name=AGENT_MEMORY_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    ai_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    logger.info("Componentes inicializados correctamente.")
except Exception as e:
    logger.error(f"Error durante la inicialización: {e}")
    sys.exit(1)

app = Server("knowledge-engine")
TRACKER_DIR.mkdir(parents=True, exist_ok=True)

def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ─── Tracker helpers ───────────────────────────────────────────────

def _tracker_path(name: str) -> Path:
    return TRACKER_DIR / name

def _read_tracker(name: str) -> dict:
    p = _tracker_path(name)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

def _write_tracker(name: str, data: dict):
    p = _tracker_path(name)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)

def _track_token_usage(project: str, feature: str, tokens: int) -> dict:
    data = _read_tracker("tokens.json")
    projects = data.setdefault("projects", {})
    proj = projects.setdefault(project, {"features": {}, "total": 0})
    proj["features"][feature] = proj["features"].get(feature, 0) + tokens
    proj["total"] += tokens
    _write_tracker("tokens.json", data)
    total = proj["total"]
    limit = 100000
    if total > limit:
        return {"alert": True, "message": f"Presupuesto excedido: {total}/{limit} tokens"}
    if total > limit * 0.8:
        return {"alert": True, "message": f"Presupuesto al {int(total/limit*100)}%: {total}/{limit} tokens", "warning": True}
    return {"alert": False, "used": total, "limit": limit, "remaining": limit - total}

def _track_failure(agent: str, project: str, feature: str, error: str, severity: str) -> dict:
    data = _read_tracker("failures.json")
    agents = data.setdefault("agents", {})
    ag = agents.setdefault(agent, [])
    entry = {
        "timestamp": now_iso(),
        "project": project,
        "feature": feature,
        "error": error,
        "severity": severity,
    }
    ag.append(entry)
    if len(ag) >= 3:
        last_3 = ag[-3:]
        if all(e["severity"] in ("alto", "crítico") for e in last_3):
            _write_tracker("failures.json", data)
            return {"alert": True, "escalate": "pm", "consecutive": len(ag[-3:])}
    _write_tracker("failures.json", data)
    return {"alert": False, "total_failures": len(ag)}

def _check_budget(project: str) -> dict:
    data = _read_tracker("tokens.json")
    proj = data.get("projects", {}).get(project)
    if not proj:
        return {"used": 0, "limit": 100000, "remaining": 100000, "ok": True, "warning": False}
    used = proj["total"]
    limit = 100000
    return {
        "used": used,
        "limit": limit,
        "remaining": limit - used,
        "ok": used <= limit,
        "warning": used > limit * 0.8,
    }

def _tracker_report(project: str | None = None) -> str:
    tokens = _read_tracker("tokens.json")
    failures = _read_tracker("failures.json")
    lines = ["# FA Tracker Report", f"Generado: {now_iso()}", ""]
    lines.append("## Tokens por proyecto")
    for pname, pdata in tokens.get("projects", {}).items():
        if project and pname != project:
            continue
        lines.append(f"- {pname}: {pdata['total']} tokens (límite: 100K)")
        for fname, ftokens in pdata.get("features", {}).items():
            lines.append(f"  - {fname}: {ftokens} tokens")
    lines.append("")
    lines.append("## Fallos por agente")
    for aname, flist in failures.get("agents", {}).items():
        lines.append(f"- {aname}: {len(flist)} fallos")
        last = flist[-1] if flist else {}
        if last:
            lines.append(f"  - Último: {last.get('severity','')} - {last.get('error','')}")
    return "\n".join(lines)

# ─── Tools ─────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="search_knowledge",
        description="Busca información en la base de conocimiento (vault gorosofia) y devuelve respuesta con DeepSeek.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Pregunta o términos de búsqueda"},
                "n_results": {"type": "integer", "description": "Máximo de fragmentos a recuperar", "default": 8}
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="store_memory",
        description="Almacena una memoria episódica o procedural de un agente en ChromaDB.",
        inputSchema={
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["episodic", "procedural"], "description": "Tipo de memoria"},
                "content": {"type": "string", "description": "Descripción del evento o procedimiento"},
                "agent_id": {"type": "string", "description": "Identificador del agente"},
                "project_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Proyecto asociado (opcional)"},
                "importance": {"type": "integer", "description": "Importancia 1-5", "minimum": 1, "maximum": 5},
                "ttl_days": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "description": "Días de vida (null = forever)"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Etiquetas para búsqueda"}
            },
            "required": ["type", "content", "agent_id"]
        }
    ),
    Tool(
        name="search_memories",
        description="Busca memorias de agentes con filtros opcionales y ordenadas por importancia.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Texto de búsqueda semántica"},
                "agent_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Filtrar por agente"},
                "project_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Filtrar por proyecto"},
                "type": {"anyOf": [{"type": "string", "enum": ["episodic", "procedural"]}, {"type": "null"}], "description": "Filtrar por tipo"},
                "min_importance": {"type": "integer", "description": "Importancia mínima (1-5)", "default": 1},
                "limit": {"type": "integer", "description": "Máximo de resultados", "default": 5}
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="forget_memory",
        description="Marca una memoria como obsoleta (soft delete).",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "UUID de la memoria a olvidar"},
                "reason": {"type": "string", "description": "Motivo del olvido"}
            },
            "required": ["memory_id", "reason"]
        }
    ),
    Tool(
        name="track_token_usage",
        description="Registra consumo de tokens de DeepSeek API por proyecto/feature.",
        inputSchema={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Nombre del proyecto"},
                "feature": {"type": "string", "description": "Feature o tarea"},
                "tokens": {"type": "integer", "description": "Tokens consumidos"}
            },
            "required": ["project", "feature", "tokens"]
        }
    ),
    Tool(
        name="track_failure",
        description="Registra un fallo de un agente en un proyecto.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": "Identificador del agente"},
                "project": {"type": "string", "description": "Nombre del proyecto"},
                "feature": {"type": "string", "description": "Feature o tarea"},
                "error": {"type": "string", "description": "Descripción del error"},
                "severity": {"type": "string", "enum": ["bajo", "medio", "alto", "crítico"], "description": "Severidad"}
            },
            "required": ["agent", "project", "feature", "error", "severity"]
        }
    ),
    Tool(
        name="check_budget",
        description="Verifica el presupuesto de tokens restante para un proyecto.",
        inputSchema={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Nombre del proyecto"}
            },
            "required": ["project"]
        }
    ),
    Tool(
        name="tracker_report",
        description="Genera un reporte de todos los proyectos activos con tokens y fallos.",
        inputSchema={
            "type": "object",
            "properties": {
                "project": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Filtrar por proyecto (opcional)"}
            }
        }
    ),
]

TOOL_MAP = {t.name: t for t in TOOLS}

async def handle_list_tools(ctx, params: PaginatedRequestParams) -> ListToolsResult:
    return ListToolsResult(tools=TOOLS)

async def handle_call_tool(ctx, params: CallToolRequestParams) -> CallToolResult:
    name = params.name
    args = params.arguments or {}
    logger.info(f"Tool called: {name}")

    try:
        if name == "search_knowledge":
            query = args.get("query", "")
            n_results = args.get("n_results", 8)
            if not query:
                return CallToolResult(content=[TextContent(type="text", text="Error: query requerido")])

            embedding = [get_embedding(query)]
            results = kb_collection.query(
                query_embeddings=embedding,
                n_results=min(n_results * 2, N_RESULTS * 2),
                include=["documents", "metadatas"]
            )

            if not results["documents"] or not results["documents"][0]:
                return CallToolResult(content=[TextContent(type="text", text="No se encontró información relevante.")])

            seen_sources = set()
            context_parts = []
            for i, doc in enumerate(results["documents"][0]):
                if len(context_parts) >= min(n_results, N_RESULTS):
                    break
                metadata = results["metadatas"][0][i]
                source = metadata.get("source", "desconocido")
                if source not in seen_sources:
                    seen_sources.add(source)
                    context_parts.append(f"[Fuente: {source}]\n{doc}")

            context = "\n\n---\n\n".join(context_parts)
            response = ai_client.chat.completions.create(
                model=DEEPSEEK_MODEL, max_tokens=DEEPSEEK_MAX_TOKENS, temperature=DEEPSEEK_TEMPERATURE,
                messages=[
                    {"role": "system", "content": "Asistente de investigación. Responde SOLO con el contexto proporcionado. Si no sabes, dilo. Cita fuentes entre [corchetes]."},
                    {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {query}"}
                ]
            )
            answer = response.choices[0].message.content
            return CallToolResult(content=[TextContent(type="text", text=answer)])

        elif name == "store_memory":
            mem_type = args.get("type", "episodic")
            content = args.get("content", "")
            agent_id = args.get("agent_id", "")
            project_id = args.get("project_id")
            importance = args.get("importance", 3)
            ttl_days = args.get("ttl_days")
            tags = args.get("tags", [])

            memory_id = str(uuid.uuid4())
            embedding = [get_embedding(content)]
            metadata = {
                "memory_id": memory_id,
                "type": mem_type,
                "agent_id": agent_id,
                "importance": str(importance),
                "timestamp": now_iso(),
                "tags": json.dumps(tags),
                "status": "active",
            }
            if project_id:
                metadata["project_id"] = project_id
            if ttl_days:
                from datetime import timedelta
                expiry = datetime.now(timezone.utc) + timedelta(days=int(ttl_days))
                metadata["ttl"] = expiry.isoformat()

            mem_collection.add(
                ids=[memory_id],
                embeddings=embedding,
                documents=[content],
                metadatas=[metadata]
            )
            logger.info(f"Memoria almacenada: {memory_id}")
            return CallToolResult(content=[TextContent(type="text", text=json.dumps({"memory_id": memory_id, "status": "stored"}))])

        elif name == "search_memories":
            query = args.get("query", "")
            agent_id = args.get("agent_id")
            project_id = args.get("project_id")
            mem_type = args.get("type")
            min_importance = args.get("min_importance", 1)
            limit = args.get("limit", 5)

            embedding = [get_embedding(query)]
            where_clause = {"status": "active"}
            if agent_id:
                where_clause["agent_id"] = agent_id
            if project_id:
                where_clause["project_id"] = project_id
            if mem_type:
                where_clause["type"] = mem_type

            results = mem_collection.query(
                query_embeddings=embedding,
                n_results=limit * 3,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            entries = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    imp = int(meta.get("importance", 0))
                    if imp < min_importance:
                        continue
                    entries.append({
                        "memory_id": meta.get("memory_id", ""),
                        "content": doc[:500],
                        "type": meta.get("type", ""),
                        "agent_id": meta.get("agent_id", ""),
                        "project_id": meta.get("project_id"),
                        "importance": imp,
                        "tags": json.loads(meta.get("tags", "[]")),
                        "timestamp": meta.get("timestamp", ""),
                    })
                    if len(entries) >= limit:
                        break

            entries.sort(key=lambda e: e["importance"], reverse=True)
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(entries, indent=2, default=str))])

        elif name == "forget_memory":
            memory_id = args.get("memory_id", "")
            reason = args.get("reason", "")
            results = mem_collection.get(where={"memory_id": memory_id})
            if results["ids"]:
                mem_collection.update(
                    ids=[memory_id],
                    metadatas=[{"status": "forgotten", "forget_reason": reason, "forgotten_at": now_iso()}]
                )
                return CallToolResult(content=[TextContent(type="text", text=f"Memoria {memory_id} marcada como olvidada.")])
            return CallToolResult(content=[TextContent(type="text", text=f"Memoria {memory_id} no encontrada.")])

        elif name == "track_token_usage":
            result = _track_token_usage(args["project"], args["feature"], args["tokens"])
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, default=str))])

        elif name == "track_failure":
            result = _track_failure(args["agent"], args["project"], args["feature"], args["error"], args["severity"])
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, default=str))])

        elif name == "check_budget":
            result = _check_budget(args.get("project", ""))
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, default=str))])

        elif name == "tracker_report":
            report = _tracker_report(args.get("project"))
            return CallToolResult(content=[TextContent(type="text", text=report)])

        else:
            return CallToolResult(content=[TextContent(type="text", text=f"Error: herramienta desconocida '{name}'")])

    except Exception as e:
        logger.error(f"Error en {name}: {e}", exc_info=True)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error interno: {str(e)}")],
            isError=True
        )

app.add_request_handler("tools/list", PaginatedRequestParams, handle_list_tools)
app.add_request_handler("tools/call", CallToolRequestParams, handle_call_tool)

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
