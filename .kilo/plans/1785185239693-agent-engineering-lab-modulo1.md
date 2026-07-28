# Plan: Iniciar Agent Engineering Lab - Módulo 1

## Contexto

Curso: **Agent Engineering Lab** en `/Users/carlosgorostizaga2/Documents/wks/gorosophia/agent-engineering-lab/`

Estructura:
- `01-harness/` → Harness Engineering
- `02-loop/` → Loop Engineering
- `03-graph/` → Graph Engineering
- `04-integration/` → Integración

Cada módulo tiene `exercises/` (ejercicios a resolver) y `examples/` (soluciones de referencia).

## Setup inicial

```bash
cd /Users/carlosgorostizaga2/Documents/wks/gorosophia/agent-engineering-lab
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejercicios del Módulo 1

### 1.1 Harness básico (`01-basic-harness.py`)

**Archivo**: `01-harness/exercises/01-basic-harness.py`
**TODO**: Implementar `Harness.execute()` (línea 60-68)

La función debe:
1. Validar que `tool_name` existe en `self.tools`
2. Medir tiempo con `time.time()`
3. Ejecutar `tool.fn(input_data)` con try/except
4. Loggear entrada (`logger.info`) y errores (`logger.error`)
5. Construir y devolver `ToolResult` (success/error, output, duration_ms)
6. Guardar cada resultado en `self.history`

**Ejecutar**: `python 01-harness/exercises/01-basic-harness.py`
**Solución referencia**: `01-harness/examples/01-basic-harness-solution.py`

### 1.2-1.4 Siguientes ejercicios

- 02-persistence.py → estado durable en JSON
- 03-permissions.py → sistema de permisos/allowlist
- 04-full-harness.py → harness completo con MCP

## Próximos módulos (tras completar 01-harness)

- `02-loop/` → ciclos de calidad, graders, reintentos
- `03-graph/` → flujos explícitos, nodos, branching
- `04-integration/` → las tres capas juntas
