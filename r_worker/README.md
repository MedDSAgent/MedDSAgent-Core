# R worker

The R tool runtime. Spawned as a subprocess by the TypeScript agent and spoken to
over the JSON-line protocol in [../docs/worker-protocol.md](../docs/worker-protocol.md).

This worker is **native R** — it does not use Python or rpy2. Running R sessions
requires no Python installation.

```
Rscript r_worker/entry.R --work_dir /path/to/session
```

| File | Role |
|------|------|
| `entry.R` | Protocol host loop: stdin/stdout JSON lines, handshake, dispatch, shutdown |
| `handler.R` | R execution: `execute`, `inject`, `get_state`, `save_state`, `load_state`, `reset_state` |

## Requirements

- **R** ≥ 4.0 (`Rscript` on `PATH`, or set `MEDDS_RSCRIPT_BIN` / the session's
  `rscript_bin` config).
- **jsonlite** — required. The protocol is JSON, and plot previews are base64-encoded
  with `jsonlite::base64_enc`. The worker exits with a clear message if it is missing.

  ```r
  install.packages("jsonlite")
  ```

- **ggplot2** — optional. Only needed if you want `ggplot` objects rendered as inline
  image previews in the variables panel. Everything else works without it.

## Notes

- **stdout is the protocol channel.** Nothing may print to it. User output is captured
  via `capture.output()`; R sends `message()`/`warning()`/package banners to stderr,
  which the parent ignores.
- **State** is an `.RData` file written with `save()` and read with `load()` — the same
  format the old rpy2-based handler used, so state saved by previous R sessions still
  loads.
- **`execute` mimics the REPL**: visible results autoprint, `invisible()` results do not.
