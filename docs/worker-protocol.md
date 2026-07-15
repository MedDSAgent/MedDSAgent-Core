# Worker IPC Protocol

This document specifies the JSON-line protocol used between the TypeScript agent
(parent process, `src/workers/`) and language-runtime worker subprocesses
(`python_worker/`, `r_worker/`, and a future `bash_worker/`).

Any language can implement a compliant worker as long as it follows this contract.

---

## 1. Transport

- **Channel**: `stdin` / `stdout` of the subprocess.
- **Framing**: newline-delimited JSON — one JSON object per line, terminated with `\n`.
- **Encoding**: UTF-8.
- **stderr** of the subprocess is captured separately by the parent and written
  to the application log; it is not part of the protocol.
- User code output (from `print()`, `cat()`, etc.) is captured **inside** the
  worker and returned as part of response payloads, never written directly to
  protocol stdout.

---

## 2. Message Shapes

### 2.1 Command (parent → worker)

```json
{ "method": "<method_name>", "params": { ... } }
```

| Field    | Type   | Required | Description                          |
|----------|--------|----------|--------------------------------------|
| `method` | string | yes      | Method name to invoke on the handler |
| `params` | object | yes      | Method arguments (can be `{}`)       |

### 2.2 Response (worker → parent)

**Success**

```json
{ "status": "ok", "data": { ... } }
```

**Error**

```json
{ "status": "error", "error": "<message or traceback>" }
```

| Field    | Type   | Description                                      |
|----------|--------|--------------------------------------------------|
| `status` | string | `"ok"` or `"error"`                             |
| `data`   | object | Present when `status == "ok"`. Method-specific. |
| `error`  | string | Present when `status == "error"`. Human-readable message or full traceback. |

---

## 3. Startup Handshake

The parent spawns the worker and reads exactly **one line** before sending any commands.

1. **Parent** spawns:
   ```
   python -m python_worker.entry python_worker.handlers.PythonHandler --work_dir /path/to/session
   ```
2. **Worker** initialises the handler.
3. **Worker** writes one line:
   - On success: `{"status": "ok", "data": <ready_info>}`
   - On failure: `{"status": "error", "error": "<traceback>"}`  
     The worker exits with code 1 after sending this.
4. **Parent** reads the line.
   - If `status == "error"`: the spawn failed; surface the error to the user.
   - If `status == "ok"`: the worker is ready to receive commands.

### `ready_info` payload (Python worker)

```json
{
  "python_version": "3.12.5",
  "available_libs": ["pandas", "numpy", "scipy"],
  "missing_optional": ["statsmodels", "matplotlib"]
}
```

### `ready_info` payload (R worker)

Spawned as `Rscript r_worker/entry.R --work_dir /path/to/session`. Note there is no
`python_version`: the R worker is native R and no Python is involved.

```json
{
  "r_version": "R version 4.5.2 (2025-10-31)",
  "available_libs": ["ggplot2", "dplyr", "tidyr", "data.table", "survival"]
}
```

---

## 4. Command Loop

After the handshake the worker enters an infinite read loop:

```
while true:
    read one line from stdin
    if EOF → exit cleanly
    parse as JSON
    if parse fails → send error response, continue
    dispatch method
    send response
```

Commands are **synchronous and serial**: the parent sends one command and waits
for its response before sending the next. There is no request-id multiplexing.

---

## 5. Built-in Methods

These methods are handled by `python_worker.entry` (the host loop) before the
handler's `dispatch()` is called.

### `shutdown`

Cleanly exit the subprocess.

**Command**: `{"method": "shutdown", "params": {}}`

**Response**: `{"status": "ok", "data": {}}`

The worker calls `handler.on_shutdown()` then exits the process after sending the
response. The parent should close the stdin pipe and `await` process exit.

---

## 6. Handler Methods

### PythonHandler (`python_worker.handlers.PythonHandler`)

#### `execute`

Run Python code in the persistent namespace.

**Command**:
```json
{ "method": "execute", "params": { "code": "<python source string>" } }
```

**Response data**:
```json
{ "output": "<captured stdout + stderr + result repr, or error>" }
```

The last expression in the code block is evaluated and its `repr` appended to
output (REPL-style). On error, output contains `[Error]\n<traceback>`.
On empty output: `"(No output)"`.

#### `inject`

Execute code silently (setup / DB injection). Output is suppressed.

**Command**:
```json
{ "method": "inject", "params": { "code": "<python source string>" } }
```

**Response data**:
```json
{ "error": null }
```
or on failure:
```json
{ "error": "<traceback string>" }
```

#### `get_state`

Return variable metadata for the session variables panel.

**Command**: `{"method": "get_state", "params": {}}`

**Response data**:
```json
{
  "variables": [
    {
      "name": "df",
      "type": "DataFrame",
      "value": "(150x5)",
      "preview": "<html table or text>",
      "is_error": false
    }
  ]
}
```

#### `save_state`

Serialize the execution namespace to disk using `dill`.

**Command**:
```json
{ "method": "save_state", "params": { "path": "/path/to/state.pkl" } }
```

**Response data**: `{}`

#### `load_state`

Deserialize a previously saved namespace from disk.

**Command**:
```json
{ "method": "load_state", "params": { "path": "/path/to/state.pkl" } }
```

**Response data**: `{}`

#### `reset_state`

Clear all user variables and re-run environment setup.

**Command**: `{"method": "reset_state", "params": {}}`

**Response data**: `{}`

---

### R worker (`r_worker/entry.R`)

Native R — no Python, no rpy2. Implements the same method set as PythonHandler
(`execute`, `inject`, `get_state`, `save_state`, `load_state`, `reset_state`) with
the same request/response shapes. Differences:

- `execute` evaluates each top-level expression in a dedicated environment and mimics
  the REPL: visible results autoprint via `print()`, `invisible()` results produce no
  output. Warnings are collected and appended under `[stderr]`; errors stop evaluation
  and are appended under `[Error]`.
- `save_state`/`load_state` use R's native `.RData` format (`save()`/`load()`) — the
  same format the previous rpy2-based handler wrote, so old state still loads.
- `get_state` walks the environment and maps R types to the same
  `{name, type, value, preview, is_error}` structure. `data.frame` previews are HTML
  tables (`class="dataframe df-table"`, cells HTML-escaped) matching what pandas'
  `to_html` produced on the Python side; `ggplot` objects render to a base64 PNG
  `<img>` tag.
- Requires the `jsonlite` R package. `ggplot2` is optional and only affects plot
  previews. See [../r_worker/README.md](../r_worker/README.md).

---

## 7. Code Safety Gate

When the environment variable `MEDDS_CODE_GATE=true` is set, both handlers
reject code that attempts to use blocked modules or calls before execution:

- **Python**: blocked via AST analysis. Blocked modules: `subprocess`, `socket`,
  `ftplib`, `smtplib`, `telnetlib`, `xmlrpc`, `socketserver`. Blocked calls:
  `os.system`, `os.popen`, `os.execv*`, `os.spawnv*`, etc.
- **R**: blocked via regex on source text. Blocked calls: `system()`,
  `system2()`, `shell()`, `shell.exec()`, `download.file()`,
  `socketConnection()`, `url()`.

On rejection the `output` field contains `[Blocked] Code contains restricted operations: ...`.

---

## 8. Cancellation

The current protocol has no in-band cancellation message. Cancellation is
implemented by the parent via:

1. Closing the worker's stdin pipe — the worker detects EOF and exits.
2. Sending `SIGTERM` / killing the subprocess if it does not exit promptly.

A future extension may add a `{"method": "cancel", "params": {}}` message for
cooperative cancellation of long-running operations.

---

## 9. Adding a New Worker Language

To add a new worker language (e.g. Bash, Julia):

1. Create a handler class that extends `WorkerHandler` (or implements the same
   interface in your language).
2. Implement `get_ready_info`, `dispatch`, and `on_shutdown`.
3. Expose an entry point that runs the same startup handshake and command loop
   as `python_worker/entry.py`.
4. Register the worker type in `src/workers/WorkerProcess.ts`.

The TS parent side does not care what language the subprocess is written in —
only that it speaks this JSON-line protocol over stdin/stdout.
