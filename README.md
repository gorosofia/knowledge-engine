# Knowledge Engine

Motor de conocimiento personal con RAG semántico y servidor MCP, optimizado para el ecosistema DeepSeek.

## Stack

| Componente | Tecnología | Coste | Motivo |
|------------|-----------|-------|--------|
| Notas | Obsidian (Markdown) | Gratis | Edición local |
| Embeddings | Ollama + nomic-embed-text | Gratis | Ya lo tienes instalado; sin API externa |
| Vector DB | ChromaDB (local) | Gratis | Sin servidor externo |
| Chat/LLM | DeepSeek V4 Flash | Muy bajo | $0.14 input / $0.28 output por 1M tokens |
| Protocolo agentes | MCP | Gratis | Estándar abierto |

## Configuración rápida

1. Copia `.env.example` a `.env` y añade tu `DEEPSEEK_API_KEY`
2. Crea un entorno virtual: `python3 -m venv venv`
3. Actívalo: `source venv/bin/activate` (Unix) o `venv\Scripts\activate` (Windows)
4. Instala dependencias: `pip install -r requirements.txt`
5. Ejecuta `python build_index.py` para indexar tus notas por Ollama
6. Consulta con `python ask.py "tu pregunta"`
7. Usa `python mcp_server.py` para exponer la herramienta MCP a agentes externos

## Optimización de costes

- Usa `deepseek-v4-flash` (modelo por defecto): es el más económico.
- Aprovecha su **contexto de 1M tokens**: puedes recuperar hasta 50+ fragmentos en una sola llamada.
- `max_tokens=1024` por defecto: limita el gasto por respuesta.
- `temperature=0`: respuestas deterministas, más cortas y predecibles.
- Los embeddings son **gratis** porque se generan en tu máquina con Ollama y `nomic-embed-text`.
- Solo pagas por las consultas que haces, no por tener el sistema encendido.

## Modelos disponibles

| Modelo | Precio input/output | Contexto | Uso recomendado |
|--------|---------------------|----------|-----------------|
| `deepseek-v4-flash` | $0.14 / $0.28 por 1M tokens | 1M | Por defecto, alto volumen |
| `deepseek-v4-pro` | $0.435 / $0.87 por 1M tokens | 1M | Razonamiento complejo |

Cambia el modelo en `.env` si necesitas más calidad.
