"""
WorkerHandler abstract base class.

Concrete subclasses live in python_worker/handlers.py.
"""

import abc
from typing import Any, Dict


class WorkerHandler(abc.ABC):
    """
    Abstract base class for subprocess worker handlers.

    The python_worker.entry host calls get_ready_info() once at startup, then
    dispatches incoming commands to dispatch() in a loop.
    """

    @abc.abstractmethod
    def get_ready_info(self) -> Dict[str, Any]:
        """
        Called once after __init__ succeeds.

        Return a JSON-serializable dict describing the worker's capabilities.
        Sent to the parent process as the startup handshake payload.

        Recommended keys (optional):
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
            Method name, e.g. "execute", "get_state", "save_state".
        params : dict
            Method arguments.

        Returns
        -------
        dict
            JSON-serializable result dict. Structure is method-specific.

        Raises
        ------
        ValueError
            For unknown method names or invalid params.
            python_worker.entry catches all exceptions and sends an error response.
        """

    def on_shutdown(self) -> None:
        """
        Called just before the subprocess exits (on "shutdown" command).
        Override to release resources (DB connections, file handles, etc.).
        """
