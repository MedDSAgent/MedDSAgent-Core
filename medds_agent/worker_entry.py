"""
General-purpose subprocess worker entry point.

This module is launched as a subprocess by CodeWorker (workers.py).
It hosts a WorkerHandler implementation that provides tool-specific logic
(Python execution, R execution, document indexing, etc.).

Launch pattern:
    python -m medds_agent.worker_entry <handler_class_path> [--key value ...]

Example:
    python -m medds_agent.worker_entry medds_agent.worker_handlers.PythonHandler --work_dir /path/to/session

Protocol:
    - Communication is via JSON lines over stdin/stdout.
    - The worker's own stdout/stderr (from user code, libraries, etc.) are
      redirected internally by the handler; this channel is reserved for protocol.
    - On startup the worker writes one "ready" or "error" JSON line, then
      enters the command loop.
    - Each command is one JSON line on stdin; each response is one JSON line on stdout.
    - The built-in "shutdown" method exits the process cleanly.

Command format:
    {"method": "<method_name>", "params": {...}}

Response format (success):
    {"status": "ok", "data": {...}}

Response format (error):
    {"status": "error", "error": "<message>"}
"""

import sys
import json
import argparse
import importlib
import traceback


def _read_line(stream) -> str:
    """Read one line from stream. Returns empty string on EOF."""
    return stream.readline()


def _write_line(stream, obj: dict):
    """Write one JSON line to stream and flush immediately."""
    stream.write(json.dumps(obj) + "\n")
    stream.flush()


def _import_class(dotted_path: str):
    """Import a class from a dotted module.ClassName path."""
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Invalid handler path '{dotted_path}'. Expected 'module.ClassName'.")
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'.")
    return cls


def main():
    # --- Reserve the real stdin/stdout for protocol before anything else ---
    proto_in = sys.stdin
    proto_out = sys.stdout

    # Redirect sys.stdout/stderr so that any stray prints from imports or
    # handler setup don't corrupt the protocol channel.
    import io
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    # --- Parse arguments ---
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("handler_class", type=str)
    args, remaining = parser.parse_known_args()

    # Convert remaining ["--key", "value", ...] into a dict
    kwargs = {}
    it = iter(remaining)
    for token in it:
        if token.startswith("--"):
            key = token[2:]
            try:
                val = next(it)
            except StopIteration:
                val = True
            kwargs[key] = val

    # --- Instantiate handler ---
    try:
        handler_cls = _import_class(args.handler_class)
        handler = handler_cls(**kwargs)
    except Exception as e:
        _write_line(proto_out, {
            "status": "error",
            "error": f"Failed to initialize handler '{args.handler_class}': {traceback.format_exc()}"
        })
        sys.exit(1)

    # --- Send ready signal ---
    try:
        ready_data = handler.get_ready_info()
    except Exception as e:
        ready_data = {}

    _write_line(proto_out, {"status": "ok", "data": ready_data})

    # --- Command loop ---
    while True:
        try:
            raw = _read_line(proto_in)
        except Exception:
            break

        if not raw:
            # EOF — parent closed the pipe
            break

        raw = raw.strip()
        if not raw:
            continue

        try:
            cmd = json.loads(raw)
        except json.JSONDecodeError as e:
            _write_line(proto_out, {"status": "error", "error": f"Invalid JSON: {e}"})
            continue

        method = cmd.get("method", "")
        params = cmd.get("params", {})

        # Built-in: shutdown
        if method == "shutdown":
            try:
                handler.on_shutdown()
            except Exception:
                pass
            _write_line(proto_out, {"status": "ok", "data": {}})
            break

        # Delegate to handler
        try:
            result = handler.dispatch(method, params)
            _write_line(proto_out, {"status": "ok", "data": result})
        except Exception as e:
            _write_line(proto_out, {
                "status": "error",
                "error": traceback.format_exc()
            })


if __name__ == "__main__":
    main()
