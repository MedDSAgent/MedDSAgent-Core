# MedDSAgent-Core — In-Process API

The package exports two top-level entry points. Choose based on your deployment target.

---

## `startServer(opts)` — HTTP mode (Docker, CLI serve)

Starts a Fastify HTTP server backed by a `SessionManager`. Returns a handle for clean shutdown.

```ts
import { startServer } from "meddsagent-core";

const server = await startServer({
  workDir: "/path/to/workspace",  // sessions live under <workDir>/sessions/
  port: 7842,                     // optional; defaults to $PORT env var or 7842
  host: "0.0.0.0",               // optional; defaults to $HOST env var or "0.0.0.0"
});

console.log(`Listening at ${server.address}`);

// Graceful shutdown:
await server.close();
```

### `ServerHandle`

| Field | Type | Description |
|-------|------|-------------|
| `address` | `string` | Full bind address, e.g. `http://0.0.0.0:7842` |
| `close` | `() => Promise<void>` | Closes the HTTP server |

---

## `createSessionManager(opts)` — In-process mode (VS Code extension)

Returns a `SessionManager` directly, with no HTTP server. The caller drives sessions by calling methods on the manager, bypassing the REST layer entirely.

```ts
import { createSessionManager } from "meddsagent-core";

const manager = createSessionManager({ workDir: "/path/to/workspace" });

// Create a session
const sessionId = await manager.createSession("My Analysis", {
  language: "python",
  llm_provider: "openai",
  llm_model: "gpt-4o",
  llm_api_key: process.env.OPENAI_API_KEY,
});

// Chat — yields SSE-style events
for await (const event of manager.chat(sessionId, "Load the CSV and describe it")) {
  switch (event.type) {
    case "response":   /* LLM text */         break;
    case "tool_running": /* tool started */   break;
    case "tool_output":  /* tool finished */  break;
    case "final_decision": /* done */         break;
  }
}

// Inspect environment
const vars = await manager.getVariables(sessionId);

// Stop a running chat
await manager.stopSession(sessionId);

// Clean up
await manager.deleteSession(sessionId);
```

### `SessionManager` API

All methods match the `ISessionManager` interface. Full type definitions are in `dist/server/types.d.ts`.

| Method | Returns | Description |
|--------|---------|-------------|
| `listSessions()` | `Promise<SessionSummary[]>` | All sessions, newest first |
| `createSession(name, config)` | `Promise<string>` | Returns session ID |
| `getSession(id)` | `Promise<SessionDetail \| null>` | Null if not found |
| `updateSession(id, name, config)` | `Promise<void>` | Config update; `language`, `python_bin`, `r_home` are immutable |
| `deleteSession(id)` | `Promise<void>` | Shuts down worker, removes files |
| `renameSession(id, name)` | `Promise<void>` | Name only |
| `getHistory(id)` | `Step[]` | All history steps (sync) |
| `chat(id, message)` | `AsyncGenerator<AgentEvent>` | Streaming agent events |
| `stopSession(id)` | `Promise<boolean>` | Aborts active chat; `false` if idle |
| `addSystemMessage(id, message)` | `Promise<void>` | Injects a system step |
| `getVariables(id)` | `Promise<VariablesSnapshot \| null>` | Live Python/R variable state |
| `getMemoryDebug(id)` | `Promise<Record<string, unknown> \| null>` | Memory window contents |
| `getCompressionDebug(id)` | `Promise<Record<string, unknown> \| null>` | Compression history |

### `AgentEvent` union

```ts
type AgentEvent =
  | { type: "response";       data: string }
  | { type: "tool_calls";     data: EnrichedToolCall[] }
  | { type: "tool_running";   toolName: string; data: Record<string, unknown>; jobId: string | null }
  | { type: "tool_output";    data: string }
  | { type: "final_decision"; isFinal: boolean };
```

### `SessionConfig` fields

| Field | Type | Notes |
|-------|------|-------|
| `language` | `"python" \| "r"` | Immutable after creation; default `"python"` |
| `llm_provider` | `string` | `"openai"` (default) or `"azure"` |
| `llm_model` | `string` | Model name / deployment |
| `llm_api_key` | `string` | API key |
| `llm_base_url` | `string` | Override base URL (for Azure endpoint or proxies) |
| `llm_api_version` | `string` | Azure API version |
| `temperature` | `number` | Default `1.0` |
| `top_p` | `number` | Default `1.0` |
| `reasoning_effort` | `"low" \| "medium" \| "high"` | Activates reasoning mode |
| `python_bin` | `string` | Path to Python executable; immutable after creation |
| `r_home` | `string` | `R_HOME` for R sessions; immutable after creation |
| `db_connection_code` | `string` | Python/R code run at startup to establish DB connection |
| `specialty_id` | `string` | ID of a specialty prompt from `/specialty-prompts` |
| `specialty_prompt` | `string` | Raw specialty prompt text |

---

## No HTTP context assumed

Importing `meddsagent-core` has no module-level side effects:

- No port is opened
- No environment variables are read at import time
- No filesystem writes occur until `createSessionManager()` or `startServer()` is called
- Specialty directory resolution is deferred until the first HTTP request

The package is safe to `import` / `await import()` in any Node.js context including VS Code extension hosts.

---

## Dynamic import from CJS (VS Code extension host)

The VS Code extension host runs CJS by default. Since `meddsagent-core` is ESM, use a dynamic import inside an async function:

```ts
// In an async function inside your CJS extension code:
const { createSessionManager } = await import("meddsagent-core");
const manager = createSessionManager({ workDir });
```

With TypeScript `"module": "node16"` (or `"node18"` / `"nodenext"`), the `await import()` expression is preserved as a real ESM import in the compiled output and works correctly against Node.js 20+.

---

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `PORT` | `7842` | HTTP port for `startServer` |
| `HOST` | `0.0.0.0` | HTTP host for `startServer` |
| `MEDDS_PYTHON_BIN` | `python` | Default Python executable for new sessions |
| `MEDDS_R_HOME` | — | Default `R_HOME` for R sessions |
| `LLM_MAX_NEW_TOKENS` | `16384` | Max tokens per LLM response |
| `LLM_MAX_INPUT_TOKENS` | `120000` | Max input context length |
| `MEMORY_START_WINDOW_SIZE` | `5` | IndexedMemory: rounds always included from start |
| `MEMORY_RECENT_WINDOW_SIZE` | `20` | IndexedMemory: recent rounds to keep uncompressed |
| `MEMORY_COMPRESS_THRESHOLD` | `2000` | IndexedMemory: chars before compression triggers |
| `MEDDS_AUTO_WAIT_TIMEOUT` | `5` | Seconds to auto-wait for async tool completion |
