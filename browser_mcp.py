#!/usr/bin/env python3
"""
MCP Server para búsqueda web.
Proporciona search_web, fetch_url, search_github, search_huggingface.
Transporte: stdio
"""

import sys, asyncio, json, logging
from urllib.parse import urlencode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stderr)
logger = logging.getLogger("browser-mcp")

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool, TextContent, CallToolResult, CallToolRequestParams,
    ListToolsResult, PaginatedRequestParams
)

app = Server("browser-mcp")

TOOLS = [
    Tool(
        name="search_web",
        description="Busca en internet usando DuckDuckGo. Devuelve título, URL y snippet de cada resultado.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Términos de búsqueda"},
                "max_results": {"type": "integer", "description": "Máximo de resultados", "default": 5}
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="fetch_url",
        description="Obtiene el contenido textual de una URL.",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL a fetchear"},
                "max_chars": {"type": "integer", "description": "Máximo de caracteres a devolver", "default": 5000}
            },
            "required": ["url"]
        }
    ),
    Tool(
        name="search_github",
        description="Busca repositorios en GitHub.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Términos de búsqueda"},
                "max_results": {"type": "integer", "description": "Máximo de resultados", "default": 5}
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="search_huggingface",
        description="Busca modelos en HuggingFace.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Términos de búsqueda"},
                "max_results": {"type": "integer", "description": "Máximo de resultados", "default": 5}
            },
            "required": ["query"]
        }
    ),
]

async def handle_list_tools(ctx, params: PaginatedRequestParams) -> ListToolsResult:
    return ListToolsResult(tools=TOOLS)

async def handle_call_tool(ctx, params: CallToolRequestParams) -> CallToolResult:
    name = params.name
    args = params.arguments or {}
    logger.info(f"Browser MCP tool: {name}")

    try:
        if name == "search_web":
            query = args["query"]
            max_results = min(args.get("max_results", 5), 20)
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            formatted = json.dumps(results, indent=2, default=str)
            return CallToolResult(content=[TextContent(type="text", text=formatted)])

        elif name == "fetch_url":
            url = args["url"]
            max_chars = args.get("max_chars", 5000)
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            text = text[:max_chars]
            return CallToolResult(content=[TextContent(type="text", text=text)])

        elif name == "search_github":
            query = args["query"]
            max_results = min(args.get("max_results", 5), 20)
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "per_page": max_results, "sort": "stars"},
                timeout=15,
                headers={"Accept": "application/vnd.github.v3+json"}
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            results = [{
                "name": r["full_name"],
                "url": r["html_url"],
                "description": r.get("description"),
                "stars": r.get("stargazers_count"),
                "language": r.get("language"),
            } for r in items]
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(results, indent=2))])

        elif name == "search_huggingface":
            query = args["query"]
            max_results = min(args.get("max_results", 5), 20)
            resp = requests.get(
                "https://huggingface.co/api/models",
                params={"search": query, "sort": "downloads", "direction": -1, "limit": max_results},
                timeout=15
            )
            resp.raise_for_status()
            items = resp.json()
            results = [{
                "modelId": m.get("modelId"),
                "pipeline_tag": m.get("pipeline_tag"),
                "downloads": m.get("downloads"),
                "likes": m.get("likes"),
            } for m in items]
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(results, indent=2))])

        else:
            return CallToolResult(content=[TextContent(type="text", text=f"Error: herramienta desconocida '{name}'")])

    except Exception as e:
        logger.error(f"Error en {name}: {e}", exc_info=True)
        return CallToolResult(content=[TextContent(type="text", text=f"Error: {str(e)}")], isError=True)

app.add_request_handler("tools/list", PaginatedRequestParams, handle_list_tools)
app.add_request_handler("tools/call", CallToolRequestParams, handle_call_tool)

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
