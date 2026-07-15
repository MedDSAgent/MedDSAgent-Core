# MedDSAgent-Core — TypeScript Rewrite TODO

This document is the implementation brief for porting MedDSAgent-Core from Python to TypeScript. It is self-contained: an agent picking this up should not need any prior conversation context.

---

## 1. Goal

Rewrite the agent backend in TypeScript so the agent runtime is **decoupled from any data-science language runtime**. Today the agent is Python, which forces every user (including R-only or chat-only users) to install Python just to run the orchestrator. After the rewrite:

- **Agent runtime** = Node/TS, bundled with the VS Code extension. No Python required.
- **Tool runtimes** = user-chosen, pluggable subprocesses (Python, R, later Bash). User installs only what they want to use.

## 2. Deployment priorities (drives every design choice)

| Target | Weight | Implication |
|--------|--------|-------------|
| VS Code extension | 60% | Agent must run **in-process** in extension host (no sidecar HTTP) |
| CLI | 20% | `npx` distribution, single command |
| Docker App | 20% | Same code, served as HTTP |

The TS code must support **both** modes from one codebase:
- `startServer(opts)` — Fastify HTTP server (Docker, CLI `serve` mode)
- `createSessionManager(opts)` — direct library API (VS Code extension imports and calls without HTTP)

## 3. Repo strategy

- **Same repo** (`MedDSAgent-Core`, GitHub: `MedDSAgent/MedDSAgent-Core`).
- The Python source under [medds_agent/](medds_agent/) stays in git history but will be **archived** at the end of the rewrite (renamed/removed in final commit; old commits remain in history).
- New TS source goes under `src/` at repo root.
- The two Python files [medds_agent/worker_entry.py](medds_agent/worker_entry.py) and [medds_agent/worker_handlers.py](medds_agent/worker_handlers.py) **stay** as the Python tool runtime — they are NOT part of the agent. Move them to `python_worker/` (or similar) and ship them as part of this repo. Same pattern will apply to a future R worker (`r_worker/`) and Bash worker.

## 4. Stack decisions (locked)

- **Node** ≥ 20, **TypeScript** strict mode
- **Fastify** for HTTP + SSE (chosen over Express for: built-in schema validation via TypeBox, async-first error handling, faster SSE primitives, ~2–3× perf)
- **better-sqlite3** for the internal DB (drop-in for current SQLite usage)
- **openai** SDK (covers OpenAI native + Azure OpenAI with one client)
- **execa** for subprocess management of language workers
- **zod** for tool-call schema validation and LLM JSON parsing
- **Vitest** for tests
- **commander** for the CLI
- **pino** for logging (Fastify default)

## 5. Architectural decisions (locked)

1. **Worker IPC stays JSON-line over stdin/stdout** — same protocol as the current Python implementation. Document the protocol explicitly in `docs/worker-protocol.md` early so any language worker (Python, R, Bash) implements the same contract.
2. **LLM engines: OpenAI-compatible only.** One `OpenAIEngine` class covers any OpenAI-compatible endpoint (native OpenAI, vLLM, SGLang, OpenRouter, and similar providers) via a configurable `baseURL`. `AzureOpenAIEngine` handles Azure OpenAI separately. No Anthropic, Ollama, or HuggingFace.
3. **IndexedMemory compression model is configurable but optional.** If the user supplies a separate compression model config, use it; otherwise compress with the same LLM the agent uses.
4. **Drop the sync agent loop.** Python has both `_process_llm_response` and `_process_llm_response_async`. TS keeps only the async + streaming generator variant.
5. **Drop DocumentSearchTool / Docling RAG from MVP.** It is rarely used in practice. Re-add later, possibly as its own subprocess tool (Python sidecar), once the core ships.
6. **Database schema: keep 1:1 with current Python SQLite schema by default.** No users to migrate, but no reason to redesign unless a specific table is awkward. If something feels wrong during port, fix it then — don't pre-redesign.
7. **REST endpoint paths stay identical** so the [MedDSAgent-App](https://github.com/MedDSAgent/MedDSAgent-App) and [MedDSAgent-VSCode](https://github.com/MedDSAgent/MedDSAgent-VSCode) clients don't need URL changes (App in particular still needs the HTTP server).

## 6. Reference: what's in the current Python codebase

~7,600 LOC total. Read these before porting each phase — they are the source of truth for behavior:

| File | LOC | Role |
|------|-----|------|
| [medds_agent/agents.py](medds_agent/agents.py) | 899 | Agent action-observation loop, streaming, auto-wait phase |
| [medds_agent/server.py](medds_agent/server.py) | 890 | FastAPI: ~30 endpoints (health, sessions, chat SSE, files, memory, variables) |
| [medds_agent/tools.py](medds_agent/tools.py) | 868 | Tool / AsyncTool base, PythonExecutor, RExecutor, FileSystem, DocumentSearch, JobWait, JobCancel, FinalResponse |
| [medds_agent/worker_handlers.py](medds_agent/worker_handlers.py) | 845 | **STAYS PYTHON** — runs inside subprocess: PythonHandler, RHandler |
| [medds_agent/memory.py](medds_agent/memory.py) | 732 | FullHistory / SlidingWindow / Indexed memory strategies |
| [medds_agent/manager.py](medds_agent/manager.py) | 624 | SessionManager: hot cache, session lifecycle, agent init |
| [medds_agent/cli.py](medds_agent/cli.py) | 480 | CLI entry points |
| [medds_agent/workers.py](medds_agent/workers.py) | 441 | Parent-side worker subprocess manager (port to TS, but protocol unchanged) |
| [medds_agent/database.py](medds_agent/database.py) | 409 | SQLite schema and access |
| [medds_agent/history.py](medds_agent/history.py) | 394 | Step types, Round, History |
| [medds_agent/job_manager.py](medds_agent/job_manager.py) | 321 | Job tracking for AsyncTool (submit, wait, cancel, status) |
| [medds_agent/document_parser.py](medds_agent/document_parser.py) | 267 | Docling-based parser (DEFER) |
| [medds_agent/engines.py](medds_agent/engines.py) | 221 | LLM clients (port only OpenAI + Azure) |
| [medds_agent/worker_entry.py](medds_agent/worker_entry.py) | 156 | **STAYS PYTHON** — subprocess host loop |
| [medds_agent/asset/prompt_templates/](medds_agent/asset/prompt_templates/) | — | Markdown prompt templates, copy verbatim |

REST endpoint inventory (from [server.py](medds_agent/server.py)):
- Health: `GET /`, `GET /health`, `POST /workspace/init`
- Specialty: `GET /specialty-prompts`, `GET /specialty-prompts/{id}`
- Sessions: `GET /sessions`, `POST /sessions`, `GET/PUT/DELETE /sessions/{id}`, `PUT /sessions/{id}/name`, `POST /test-db-connection`
- Chat: `GET /sessions/{id}/history`, `POST /sessions/{id}/chat` (SSE), `POST /sessions/{id}/stop`
- Environment: `GET /sessions/{id}/variables`
- Memory: `GET /sessions/{id}/memory`, `GET /sessions/{id}/memory/compression`
- Files: `GET/POST /sessions/{id}/files`, `POST /sessions/{id}/index`, `GET /sessions/{id}/files/{name}/index-status`, `GET/PUT /sessions/{id}/files/{path}/content`, `GET /sessions/{id}/files/{path}`, `DELETE /sessions/{id}/files/{path}`

---

## 7. Phased TODO

Each phase ends with a working, tested checkpoint. Do not skip ahead — later phases depend on earlier types.

### Phase 0 — Scaffold

- [ ] Initialize `package.json`, `tsconfig.json` (strict mode), `.gitignore` additions for `node_modules/`, `dist/`
- [ ] Set up Vitest, ESLint, Prettier
- [ ] Create directory layout:
  ```
  src/
    history/
    memory/
    engines/
    agents/
    tools/
    workers/
    server/
    session/
    cli/
    db/
    util/
  python_worker/         # moved from medds_agent/{worker_entry,worker_handlers}.py
  prompts/               # moved from medds_agent/asset/prompt_templates/
  tests/
  docs/
  ```
- [ ] Move [medds_agent/worker_entry.py](medds_agent/worker_entry.py) and [medds_agent/worker_handlers.py](medds_agent/worker_handlers.py) into `python_worker/` as a small standalone package. Update its module paths and ensure it still runs via `python -m python_worker.entry <HandlerClass> --work_dir ...`.
- [ ] Move [medds_agent/asset/prompt_templates/](medds_agent/asset/prompt_templates/) → `prompts/`. Build process should copy them into `dist/prompts/`.
- [ ] Write `docs/worker-protocol.md` documenting the JSON-line IPC contract by reading the current Python implementation. Include: startup handshake (`get_ready_info`), command dispatch shape, response shape, error shape, cancellation.
- [ ] CI: run `tsc --noEmit` + `vitest run` on push.

### Phase 1 — Foundations (no LLM yet)

- [ ] Port [history.py](medds_agent/history.py) → `src/history/`. Step union (`UserStep | AgentStep | ObservationStep | SystemStep`), `Round`, `History` class. JSON serialization round-trip tested.
- [ ] Port [database.py](medds_agent/database.py) → `src/db/`. Same SQLite schema. Wrap better-sqlite3. Test: create session, write/read history, write/read config.
- [ ] Port [engines.py](medds_agent/engines.py) → `src/engines/`. Implement `OpenAIEngine` (covers any OpenAI-compatible endpoint: OpenAI, vLLM, SGLang, OpenRouter, etc.) and `AzureOpenAIEngine`. Common interface with `chat(messages, tools)` returning `{ response, tool_calls }`. Tool-call argument normalization (LLMs sometimes return strings vs objects — match Python behavior).
- [ ] Port prompt template loader (current Python uses `importlib.resources`). TS equivalent: read from `prompts/` dir relative to package root, with a fallback for dev vs bundled.

### Phase 2 — Memory (the differentiator)

- [ ] Port `AgentMemory` base class
- [ ] Port `FullHistoryAgentMemory`
- [ ] Port `SlidingWindowAgentMemory` (start_window_size + end_window_size)
- [ ] Port `IndexedAgentMemory` — the LLM-compression strategy. Compression model: read from a separate optional config slot; if absent, fall back to the agent's main LLM engine.
- [ ] Port `get_memory_debug` and `get_compression_debug` (the API exposes these).
- [ ] Tests: known step sequences → expected `messages` arrays; compression triggers at the right thresholds.

### Phase 3 — Agent loop

- [ ] Port `JobManager` from [job_manager.py](medds_agent/job_manager.py) — job statuses (`COMPLETED | FAILED | TIMED_OUT | CANCELLED`), `submit`, `wait_async`, `cancel`, `has_pending`.
- [ ] Port `Tool` and `AsyncTool` abstract classes.
- [ ] Port `FinalResponseTool`, `JobWaitTool`, `JobCancelTool` (these are language-runtime-free, pure orchestration tools — auto-injected by the agent).
- [ ] Port `Agent` class. Single async streaming generator (no sync variant). Yields events: `tool_running`, `tool_output`, `response`, `final_decision`. Implement Phase 1 dispatch + Phase 2 auto-wait pattern from [agents.py](medds_agent/agents.py:247-320). Auto-wait timeout from `MEDDS_AUTO_WAIT_TIMEOUT` env var (default 5s).
- [ ] Reproduce all the round-validation rules: must always call a tool, `final_response` must be alone, cannot finalize while async job pending.
- [ ] Tests with a mock engine that scripts tool-call sequences.

### Phase 4 — Tool runtime via subprocess (the decoupling payoff)

- [ ] Port `WorkerHandler`-parent-side from [workers.py](medds_agent/workers.py) → `src/workers/WorkerProcess.ts`. Use `execa`. Implement: spawn, startup handshake, JSON-line read/write loop, command dispatch, cancellation, restart.
- [ ] Implement `PythonExecutorTool` (TS class). It wraps a `WorkerProcess` running `python -m python_worker.entry python_worker.handlers.PythonHandler --work_dir ...`. Interpreter path comes from session config (`pythonPath`).
- [ ] Implement `RExecutorTool` similarly (deferred until R worker added).
- [ ] Implement `FileSystemTool` (sync, in-process — no subprocess needed).
- [ ] Validate the Python worker still works unchanged after Phase 0 move. End-to-end test: TS parent spawns Python worker, sends `execute({code: "print(1+1)"})`, gets back `2`.
- [ ] Add a no-tool config path: agent can be created with empty tool list (chat-only mode). Verify nothing requires Python on disk.
- [ ] VS Code helper: read interpreter path from VS Code's Python extension settings (`python.defaultInterpreterPath`) when running in extension host.

### Phase 5 — REST API

- [ ] Set up Fastify with TypeBox schemas.
- [ ] Port endpoints from [server.py](medds_agent/server.py) one tag at a time, in this order: Health → Specialty → Sessions → Chat (SSE) → Memory → Variables → Files. Keep paths and request/response shapes identical.
- [ ] SSE for `POST /sessions/{id}/chat`: stream the agent's generator events.
- [ ] Multipart upload for `POST /sessions/{id}/files`.
- [ ] CORS open (matches current `allow_origins=["*"]`).
- [ ] Smoke test against current MedDSAgent-App docker-compose by swapping the backend image.

### Phase 6 — Session manager

- [ ] Port [manager.py](medds_agent/manager.py) → `src/session/SessionManager.ts`. Hot cache of `{ agent, worker, last_active, persisted_steps }`. Session create/delete/update. Agent instantiation from stored config. System-step injection on DB connection. Active-task tracking for cancellation.
- [ ] Per-session work_dir (`workspace/sessions/{session_id}/{uploads,outputs,scripts,internal}`).
- [ ] Persistence: agent state, history, memory, worker variable state (via the worker's `save_state`/`load_state` Python-side methods — already implemented).

### Phase 7 — VS Code in-process mode

- [ ] Refactor entry points so the package exports both:
  - `startServer(opts)` — Fastify HTTP server
  - `createSessionManager(opts)` — direct library API for VS Code extension
- [ ] Ensure no module-level side effects that assume HTTP context.
- [ ] Document the in-process API in `docs/in-process-api.md` for the VS Code extension to consume.

### Phase 8 — CLI

- [ ] Build a `commander`-based CLI: `meddsagent serve [--port]`, `meddsagent session list`, `meddsagent session new`, etc. — match current [cli.py](medds_agent/cli.py) surface.
- [ ] Publishable as `npx meddsagent ...`.

### Phase 9 — Cutover

- [ ] Update [Dockerfile](Dockerfile) to build the TS image (Node base, copy `dist/` and `python_worker/`, install Python only if user wants Python tools — make it an optional layer).
- [ ] Update [MedDSAgent-App](https://github.com/MedDSAgent/MedDSAgent-App) docker-compose to point at the new image.
- [ ] Update [MedDSAgent-VSCode](https://github.com/MedDSAgent/MedDSAgent-VSCode) extension to consume the in-process library.
- [ ] Final commit: delete `medds_agent/` Python sources (history is preserved in git). Update root [README.md](README.md) to reflect the new architecture. Add a "Legacy Python implementation" note pointing readers to the last commit on the Python branch (tag it `python-final` before deletion).
- [ ] `pyproject.toml`, `medds_agent.egg-info/`, `build/` — remove.

---

## 8. Out of scope (do NOT do)

- Anthropic, Ollama, HuggingFace LLM engines (not OpenAI-compatible; TS port deliberately drops them)
- Docling document parser / DocumentSearchTool / RAG indexing — defer to a post-MVP phase as a separate subprocess tool
- Any new features. This is a port, not a redesign. If something looks improvable, note it as a follow-up but ship the port first.
- Don't preserve backwards compatibility with the Python codebase (no users yet).

## 9. Definition of done

- All current REST endpoints respond with the same payloads from the TS server
- A session can be created, chat works end-to-end, Python tool execution works via the (unchanged) Python worker subprocess
- The agent runs without Python installed if no Python-using tools are configured
- VS Code extension can `import` the package and run an agent without spawning a sidecar HTTP server
- Docker App works against the new image with no client changes
- Python sources are removed from `main`; final Python state is preserved at tag `python-final`
