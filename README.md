# MedDSAgent-Core

The orchestration backend for MedDSAgent, a medical data science agent.

The agent runtime is TypeScript and has no data-science language runtime of its
own. Language execution happens in **pluggable subprocess workers**, so a user
installs only the runtime they actually intend to use — and a chat-only user
installs none at all. Python is not required to run the agent.

## Architecture

```
┌──────────────────────────────────────────┐
│ Agent runtime (Node / TypeScript)        │
│   history · memory · engines · agent     │
│   loop · jobs · session manager          │
└───────────────┬──────────────────────────┘
                │ JSON-line IPC over stdin/stdout
                │ (docs/worker-protocol.md)
    ┌───────────┴────────────┐
    │                        │
┌──────────────┐   ┌──────────────┐
│ python_worker│   │ r_worker     │
│ python -m …  │   │ Rscript …    │
└──────────────┘   └──────────────┘
```

Each worker is native to its own language: the R worker is plain R, so R users
need no Python, and Python users need no R.

The same codebase serves three targets:

| Target | Entry point |
|--------|-------------|
| VS Code extension | `createSessionManager(opts)` — in-process library, no HTTP sidecar |
| Docker / server | `startServer(opts)` — Fastify HTTP + SSE |
| CLI | `meddsagent serve` / `meddsagent chat` |

## Requirements

- **Node ≥ 24.** The internal database uses the built-in `node:sqlite`, which is
  only available unflagged from Node 24 onward. There are no native addons and no
  build toolchain needed.
- **Python** only if you use the Python executor tool. Install
  `python_worker/requirements.txt` into whichever interpreter you point the agent at.
- **R** only if you use the R executor tool: R with the `jsonlite` package, and
  `Rscript` on `PATH` (or set `MEDDS_RSCRIPT_BIN`). See
  [r_worker/README.md](r_worker/README.md).

Neither is needed for chat-only use, and neither is needed to use the other.
The Docker image ships both (see below), so this only applies to local installs.

## Quick start

```bash
npm install
npm run build
node dist/cli/index.js serve --work-dir ./workspace --port 7842
```

See [run_note.md](run_note.md) for the interactive chat invocation.

## Docker

```bash
docker build -t meddsagent-core .
docker run -p 7842:7842 -v /path/to/workspace:/workspace meddsagent-core
```

The image ships **both** language runtimes, so `language: "python"` and
`language: "r"` both work out of the box:

| | Version | Packages |
|---|---|---|
| Node | 24 | the agent runtime |
| Python | 3.11 | pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn, SQLAlchemy, openpyxl, pyarrow, … |
| R | 4.6 (from CRAN's Debian repo) | jsonlite, ggplot2, dplyr, tidyr, data.table, survival + recommended |

The image contains **no C/C++ compiler**: every Python and R dependency installs
from a prebuilt binary (pip wheels; [Posit Package Manager](https://packagemanager.posit.co)
binaries for R). Both installs are pinned binary-only, so the build fails loudly
rather than silently pulling a ~330MB toolchain back into the runtime image.

## Docs

- [docs/in-process-api.md](docs/in-process-api.md) — the library API the VS Code extension consumes
- [docs/worker-protocol.md](docs/worker-protocol.md) — the JSON-line IPC contract every language worker implements

## Development

```bash
npm run typecheck   # tsc --noEmit
npm test            # vitest run
npm run lint
```

## Legacy Python implementation

This project was originally written in Python. That implementation has been
removed from `main` and is preserved at the **`python-final`** tag:

```bash
git show python-final:medds_agent/agents.py
git checkout python-final    # browse the full Python tree
```

The rewrite deliberately dropped some things that existed in the Python version:
the Anthropic / Ollama / HuggingFace LLM engines, and the docling-based document
parser with its `DocumentSearch` RAG tool. LLM support is now OpenAI-compatible
endpoints (via `--base-url`) plus Azure OpenAI. Document RAG is deferred and will
return as its own subprocess worker rather than as agent-side code; the last
Python implementation of it is at `python-final`.
