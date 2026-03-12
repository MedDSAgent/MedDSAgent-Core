"""
Worker process management.

This module provides:

1. WorkerHandler (abstract) — base class for all subprocess handler implementations.
   Concrete subclasses live in worker_handlers.py.

2. CodeWorker — parent-side manager for a single subprocess worker.
   Handles spawning, the startup handshake, JSON-line communication,
   cancellation, and restart.
"""

import abc
import sys
import json
import asyncio
import subprocess
import threading
import traceback
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WorkerHandler — abstract base (runs INSIDE the subprocess)
# ---------------------------------------------------------------------------

class WorkerHandler(abc.ABC):
    """
    Abstract base class for subprocess worker handlers.

    Subclasses implement tool-specific logic (Python execution, R execution,
    document indexing, etc.) and run inside the worker subprocess.

    The worker_entry.py host calls get_ready_info() once at startup, then
    dispatches incoming commands to dispatch() in a loop.
    """

    @abc.abstractmethod
    def get_ready_info(self) -> Dict[str, Any]:
        """
        Called once after __init__ succeeds.

        Return a JSON-serializable dict describing the worker's capabilities.
        This is sent to the parent as the startup handshake payload.

        Recommended keys (optional but useful):
            available_libs: List[str]   — libraries confirmed importable
            python_version: str         — e.g. "3.11.5"
            missing_optional: List[str] — optional packages not found
        """

    @abc.abstractmethod
    def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a method call to the handler's implementation.

        Parameters
        ----------
        method : str
            Method name (e.g. "execute", "get_state", "save_state").
        params : dict
            Method arguments.

        Returns
        -------
        dict
            JSON-serializable result dict. Structure is method-specific.

        Raises
        ------
        ValueError
            If the method is unknown or params are invalid.
            worker_entry.py catches all exceptions and returns an error response.
        """

    def on_shutdown(self):
        """
        Called just before the subprocess exits (on "shutdown" command).
        Override to release resources (DB connections, file handles, etc.).
        """


# ---------------------------------------------------------------------------
# CodeWorker — parent-side subprocess manager
# ---------------------------------------------------------------------------

class WorkerStartupError(Exception):
    """Raised when the worker subprocess fails to start or reports an error during handshake."""


class WorkerDeadError(Exception):
    """Raised when a command is sent to a worker that has exited unexpectedly."""


class CodeWorker:
    """
    Parent-side manager for a single long-lived subprocess worker.

    Responsibilities:
    - Spawn the subprocess using the configured Python binary.
    - Perform the startup handshake (read the ready/error message).
    - Send JSON commands and receive JSON responses.
    - Track whether a job is currently running (one at a time).
    - Cancel a running job by restarting the subprocess (state is preserved
      via save/load state before and after when possible, but for a hard
      cancel, state from the cancelled job is lost).
    - Provide both sync and async interfaces for send_command().

    Thread safety:
    - send_command_sync() uses a threading.Lock.
    - send_command_async() uses an asyncio.Lock.
    - Do not mix sync and async calls on the same instance.
    """

    def __init__(
        self,
        handler_class_path: str,
        python_bin: Optional[str] = None,
        handler_kwargs: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        handler_class_path : str
            Dotted path to the WorkerHandler subclass, e.g.
            "medds_agent.worker_handlers.PythonHandler".
        python_bin : str, optional
            Path to the Python interpreter to use.  Defaults to sys.executable
            (same interpreter as the server).
        handler_kwargs : dict, optional
            Key-value pairs passed as --key value CLI arguments to the handler.
            Example: {"work_dir": "/path/to/session"}.
        env : dict, optional
            Extra environment variables to inject into the subprocess.
            Merged on top of the current process environment.
            Example: {"R_HOME": "/path/to/conda/envs/r_env/lib/R"}.
        """
        self.handler_class_path = handler_class_path
        self.python_bin = python_bin or sys.executable
        self.handler_kwargs = handler_kwargs or {}
        self.env = env or {}

        self._process: Optional[subprocess.Popen] = None
        self._ready_info: Dict[str, Any] = {}
        self._dead = False

        self._sync_lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None  # created lazily

        # Start the subprocess and perform handshake
        self._start()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def ready_info(self) -> Dict[str, Any]:
        """Metadata returned by the handler during the startup handshake."""
        return self._ready_info

    @property
    def is_alive(self) -> bool:
        """True if the subprocess is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _build_cmd(self) -> List[str]:
        cmd = [self.python_bin, "-m", "medds_agent.worker_entry", self.handler_class_path]
        for key, value in self.handler_kwargs.items():
            cmd += [f"--{key}", str(value)]
        return cmd

    def _start(self):
        """Spawn subprocess and read the startup handshake."""
        cmd = self._build_cmd()
        logger.debug("CodeWorker: spawning %s", cmd)

        proc_env = None
        if self.env:
            import os as _os
            proc_env = {**_os.environ, **self.env}

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            env=proc_env,
        )

        # Read startup message (one JSON line)
        try:
            raw = self._process.stdout.readline()
        except Exception as e:
            raise WorkerStartupError(f"Failed to read startup message: {e}")

        if not raw:
            stderr_output = self._process.stderr.read()
            raise WorkerStartupError(
                f"Worker subprocess exited immediately.\nstderr:\n{stderr_output}"
            )

        try:
            msg = json.loads(raw.strip())
        except json.JSONDecodeError:
            raise WorkerStartupError(f"Invalid startup JSON from worker: {raw!r}")

        if msg.get("status") != "ok":
            raise WorkerStartupError(
                f"Worker reported startup error: {msg.get('error', 'unknown error')}"
            )

        self._ready_info = msg.get("data", {})
        self._dead = False
        logger.debug("CodeWorker: ready. info=%s", self._ready_info)

    # ------------------------------------------------------------------
    # Low-level communication
    # ------------------------------------------------------------------

    def _send_recv(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send one command and read one response (blocking).
        Caller must hold _sync_lock.
        """
        if self._dead or not self.is_alive:
            raise WorkerDeadError("Worker subprocess is not running.")

        payload = json.dumps({"method": method, "params": params}) + "\n"

        try:
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except BrokenPipeError:
            self._dead = True
            raise WorkerDeadError("Worker stdin pipe is broken (process likely died).")

        try:
            raw = self._process.stdout.readline()
        except Exception as e:
            self._dead = True
            raise WorkerDeadError(f"Failed to read worker response: {e}")

        if not raw:
            self._dead = True
            stderr_output = ""
            try:
                stderr_output = self._process.stderr.read()
            except Exception:
                pass
            raise WorkerDeadError(
                f"Worker process exited unexpectedly.\nstderr:\n{stderr_output}"
            )

        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            raise WorkerDeadError(f"Worker returned invalid JSON: {raw!r}. Error: {e}")

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def send_command_sync(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send a command to the worker and return the response (blocking).

        Parameters
        ----------
        method : str
        params : dict, optional
        timeout : float, optional
            If provided, raises TimeoutError if no response within this many seconds.

        Returns
        -------
        dict
            The "data" field of the response on success.

        Raises
        ------
        WorkerDeadError
        WorkerStartupError
        TimeoutError
        RuntimeError  — if the worker returns status "error"
        """
        if params is None:
            params = {}

        with self._sync_lock:
            if timeout is not None:
                result_holder = {}
                error_holder = {}

                def _do():
                    try:
                        result_holder["v"] = self._send_recv(method, params)
                    except Exception as e:
                        error_holder["e"] = e

                t = threading.Thread(target=_do, daemon=True)
                t.start()
                t.join(timeout)

                if t.is_alive():
                    # Timed out — mark dead so next call forces restart check
                    self._dead = True
                    raise TimeoutError(
                        f"Worker did not respond to '{method}' within {timeout}s."
                    )

                if "e" in error_holder:
                    raise error_holder["e"]

                response = result_holder["v"]
            else:
                response = self._send_recv(method, params)

        if response.get("status") != "ok":
            raise RuntimeError(
                f"Worker returned error for '{method}': {response.get('error', 'unknown')}"
            )

        return response.get("data", {})

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def send_command_async(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Async version of send_command_sync.
        Runs the blocking I/O in a thread pool to avoid blocking the event loop.
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        if params is None:
            params = {}

        async with self._async_lock:
            try:
                coro = asyncio.to_thread(self._send_recv, method, params)
                if timeout is not None:
                    response = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    response = await coro
            except asyncio.TimeoutError:
                self._dead = True
                raise TimeoutError(
                    f"Worker did not respond to '{method}' within {timeout}s."
                )

        if response.get("status") != "ok":
            raise RuntimeError(
                f"Worker returned error for '{method}': {response.get('error', 'unknown')}"
            )

        return response.get("data", {})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def restart(self):
        """
        Force-kill the current subprocess and start a fresh one.
        Used after a cancel() to get a clean worker immediately.

        Unlike shutdown(), this does NOT send a graceful "shutdown" command
        (which would block waiting for the _sync_lock held by the job thread).
        Instead it terminates the process directly so the job thread unblocks,
        then starts a new subprocess.

        Note: in-flight state from the killed job is lost.
        """
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            except Exception:
                pass
        self._dead = True
        self._process = None
        self._start()

    def shutdown(self, wait: bool = True):
        """
        Gracefully shut down the subprocess.
        Sends "shutdown" command, then terminates if it doesn't exit.
        """
        if self._process is None:
            return

        if self.is_alive:
            try:
                with self._sync_lock:
                    payload = json.dumps({"method": "shutdown", "params": {}}) + "\n"
                    self._process.stdin.write(payload)
                    self._process.stdin.flush()
            except Exception:
                pass

            if wait:
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._process.kill()

        self._dead = True
        self._process = None
