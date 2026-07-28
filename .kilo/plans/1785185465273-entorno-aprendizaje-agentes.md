# Plan: Entorno de Aprendizaje — Configuración y Pruebas de Agentes

## Objetivo

Crear un entorno seguro (sandbox) sobre el curso Agent Engineering Lab donde el usuario aprenda a **configurar agentes reales** y **escribir tests** para entender su comportamiento, usando su propio vault de conocimiento (`knowledge-vault`) como dominio de prueba vía MCP.

La progresión es pedagógica: **desde el protocolo MCP hasta sistemas multi-capa**, todo construido desde cero, sin frameworks.

## Stack técnico

- Python 3.12, Apple Silicon (arm64)
- `knowledge-engine/venv` — venv con MCP SDK, ChromaDB, OpenAI, Ollama
- `mcp_server.py` — expone `search_knowledge` vía MCP stdio (ya funciona)
- `knowledge-vault/` — bóveda Obsidian indexada en ChromaDB

## Fases de implementación

---

### Fase 0: Setup del entorno

**Archivos**: `agent-engineering-lab/`
- Crear `venv/` con `python3 -m venv venv`
- Instalar dependencias: `openai`, `python-dotenv`, `requests`, `mcp>=1.0.0`, `chromadb>=0.4.0`, `pypdf`, `markdown`, `numpy`, `tqdm`
- Copiar `.env` desde `knowledge-engine/.env` (o crear uno propio con las mismas variables)
- Verificar que `knowledge-engine/build_index.py` tiene el vault indexado en ChromaDB
- Verificar que `mcp_server.py` arranca y responde

**Verificación**: `python3 -c "from mcp import Server; print('ok')"` y `curl localhost:11434/api/embeddings` (Ollama)

---

### Fase 1: MCP Harness (Módulo 1, Ejercicio 2)

**Archivo nuevo**: `01-harness/exercises/02-mcp-harness.py`

Harness que se conecta al MCP del knowledge-engine como subproceso stdio. El alumno implementa:

1. **MCPClient** — clase que:
   - Lanza `mcp_server.py` como subproceso (`Popen` con `stdin=PIPE, stdout=PIPE`)
   - Envía mensajes JSON-RPC (líneas separadas por `\n`)
   - Lee respuestas (una línea por mensaje)
   - Implementa handshake: `initialize` → `initialized` → `tools/list` → `tools/call`
   - Maneja ciclo de vida: `start()`, `stop()`, `__enter__`/`__exit__`

2. **Harness (MCP-aware)** — reusa el mismo patrón del ejercicio 1 pero:
   - `register_tool()` recibe `MCPClient` como tool
   - `execute()` llama al MCP vía `tools/call`
   - Timeout vía `signal.alarm()` o `asyncio.wait_for()`

3. **Tests**:
   - Conectar al MCP, listar tools, verificar que `search_knowledge` existe
   - Llamar `search_knowledge` con query real, verificar respuesta no vacía
   - Llamar con tool inexistente, verificar error
   - Verificar que `history` registra todos los calls
   - Verificar que el subproceso se cierra limpiamente al final

**Referencia**: `01-harness/examples/02-mcp-harness-solution.py`

---

### Fase 2: Persistencia de sesión (Módulo 1, Ejercicio 3)

**Archivo nuevo**: `01-harness/exercises/03-persistence.py`

Extiende el MCP Harness con persistencia a JSON:

1. **SessionManager**:
   - `save_session(path, history)` — exporta `history` a JSON con timestamp
   - `load_session(path)` — carga historial previo
   - `list_sessions(sessions_dir)` — lista sesiones disponibles

2. **Cambios en Harness**:
   - `execute()` llama `save_session` automáticamente tras cada tool call
   - `restore_session(path)` para cargar historial anterior
   - Formato: `sessions/{YYYY-MM-DD}/{HH-MM-SS}.json`

3. **Tests**:
   - Ejecutar tool, verificar que se crea archivo JSON
   - Cargar sesión, verificar que `history` contiene las entradas anteriores
   - Sesión vacía no falla
   - Múltiples sesiones no se sobreescriben
   - JSON es legible y tiene estructura correcta (ToolResult serializado)

---

### Fase 3: Permisos y auditoría (Módulo 1, Ejercicio 4)

**Archivo nuevo**: `01-harness/exercises/04-permissions.py`

Añade capa de seguridad sobre el MCP Harness:

1. **PermissionManager**:
   - `allowlist: list[str]` — queries permitidas (regex o palabras clave)
   - `denylist: list[str]` — queries prohibidas
   - `audit_log: list[dict]` — registro timestamp + query + permitido/denegado
   - `check_query(query) -> bool` — evalúa allow/deny
   - `log_query(query, allowed, reason)` — audita

2. **Cambios en Harness**:
   - `execute()` consulta PermissionManager antes de llamar al MCP
   - Query denegada → ToolResult error sin llamar al MCP
   - Query permitida → ToolResult normal + auditoría

3. **Tests**:
   - Query en allowlist → pasa
   - Query en denylist → bloqueada
   - Query sin allowlist configurada → comportamiento definido (allow o deny por defecto)
   - Audit log contiene todas las queries (permitidas y denegadas)
   - Audit log sobrevive a reinicio (persistencia JSON)

---

### Fase 4: Sandbox de experimentación

**Archivo nuevo**: `01-harness/sandbox/README.md`

Directorio para que el usuario experimente libremente:

- `sandbox/config.yaml` — template de configuración (tools, permisos, modelo)
- `sandbox/experiments/` — cada experimento es un `.py` autocontenido
- `sandbox/run_test.py` — script helper para ejecutar experimentos con assertions
- Instrucciones: cómo modificar config, qué parámetros tocar, cómo medir resultados

El sandbox NO tiene ejercicios — es un patio de recreo.

---

### Fase 5: Módulo 2 — Loop Engineering (roadmap)

Ejercicios que construyen sobre el MCP Harness:

- **01-simple-loop.py**: Query al vault → evaluar si respuesta es útil → requery si no
- **02-verification-loop.py**: Grader que verifica respuesta contra el vault
- **03-budget-loop.py**: Máximo de tokens, máximo de queries al vault
- **04-stacked-loops.py**: Loop de verificación dentro de loop de mejora

### Fase 6: Módulo 3 — Graph Engineering (roadmap)

- **01-linear-graph.py**: search → extract → summarize (3 nodos secuenciales)
- **02-conditional-graph.py**: vault tiene resultado → deep dive, no → web fallback
- **03-parallel-graph.py**: 3 queries paralelas al vault → merge resultados
- **04-graph-with-loops.py**: grafo con ciclos de verificación + escalada humana

### Fase 7: Módulo 4 — Integración (roadmap)

- **01-research-agent.py**: harness + MCP vault
- **02-add-loop.py**: research + verification loop
- **03-full-system.py**: harness + loop + graph + MCP + persistencia + permisos

---

## Estructura final del proyecto

```
agent-engineering-lab/
├── .env                        ← API keys (copiar de knowledge-engine)
├── venv/                       ← Python virtualenv
├── requirements.txt            ← dependencias
│
├── 01-harness/
│   ├── README.md               ← teoría (ya existe)
│   ├── exercises/
│   │   ├── 01-basic-harness.py  ← COMPLETADO (tools simuladas)
│   │   ├── 02-mcp-harness.py    ← FASE 1: MCP subprocess
│   │   ├── 03-persistence.py    ← FASE 2: sesiones JSON
│   │   └── 04-permissions.py    ← FASE 3: allowlist + auditoría
│   ├── examples/
│   │   ├── 01-basic-harness-solution.py  ← existe
│   │   ├── 02-mcp-harness-solution.py    ← FASE 1
│   │   ├── 03-persistence-solution.py    ← FASE 2
│   │   └── 04-permissions-solution.py    ← FASE 3
│   └── sandbox/                ← FASE 4: zona libre
│       ├── README.md
│       ├── config.yaml
│       └── experiments/
│
├── 02-loop/                    ← FASE 5 (roadmap)
├── 03-graph/                   ← FASE 6 (roadmap)
└── 04-integration/             ← FASE 7 (roadmap)
```

## Dependencias

```txt
openai>=1.0.0
python-dotenv>=1.0.0
requests>=2.31.0
mcp>=1.0.0
chromadb>=0.4.0
pypdf>=3.0.0
markdown>=3.4.0
numpy>=1.24.0
tqdm>=4.65.0
```

## Convenciones de código

- Tests al final de cada ejercicio (como `01-basic-harness.py`)
- Solución en `examples/` con `-solution` suffix
- Type hints Python 3.12+ (`| None`, `list[str]`, etc.)
- Logging con `logging.basicConfig`, formato `%(asctime)s - %(message)s`
- Cada test imprime `✅ Test N passed: ...` y `🎉 Todos los tests pasaron!`
- Sin dependencias externas más allá de las listadas
- `if __name__ == "__main__":` ejecuta los tests

## Validación

Cada ejercicio se valida ejecutando:
```bash
cd agent-engineering-lab
python 01-harness/exercises/02-mcp-harness.py   # Fase 1
python 01-harness/exercises/03-persistence.py    # Fase 2
python 01-harness/exercises/04-permissions.py    # Fase 3
```

Requisitos para que los tests pasen:
- Ollama corriendo localmente con `nomic-embed-text`
- `build_index.py` ejecutado (ChromaDB con datos)
- `.env` con `DEEPSEEK_API_KEY` válida
- `knowledge-engine` no necesita estar corriendo — el harness lanza su propio MCP
