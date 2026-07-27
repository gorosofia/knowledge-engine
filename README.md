# Knowledge Engine

Motor de indexación semántica, consulta RAG y servidor MCP para la base de conocimiento personal de Gorosophia.

## Contenido

- `build_index.py` - Indexa documentos (txt, md, pdf) en ChromaDB usando embeddings de BAAI/bge-m3
- `ask.py` - Consulta la base de conocimiento y genera respuestas con DeepSeek
- `mcp_server.py` - Servidor MCP que expone la herramienta `search_knowledge` para agentes externos
- `requirements.txt` - Dependencias Python
- `.env.example` - Plantilla de variables de entorno

## Configuración rápida

1. Copia `.env.example` a `.env` y añade tu `DEEPSEEK_API_KEY`
2. Crea un entorno virtual: `python3 -m venv venv`
3. Actívalo: `source venv/bin/activate` (Unix) o `venv\Scripts\activate` (Windows)
4. Instala dependencias: `pip install -r requirements.txt`
5. Ejecuta `python build_index.py` para indexar tus notas
6. Consulta con `python ask.py "tu pregunta"`
7. Usa `python mcp_server.py` para exponer la herramienta MCP a agentes externos

## Documentación

Ver la guía completa en el repositorio principal de la organización.
