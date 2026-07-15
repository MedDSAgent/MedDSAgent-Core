"""
WorkerHandler implementations that run inside subprocess workers.

Each class is launched by python_worker.entry via:
    python -m python_worker.entry python_worker.handlers.<ClassName> --work_dir /path

Available handlers:
    PythonHandler   — persistent Python execution environment

R is not handled here. It has its own native worker in r_worker/ (Rscript), so
R users need no Python installation at all.
"""

import ast
import os
import sys
import io
import json
import base64
import traceback
from typing import Any, Dict, List, Optional

from python_worker.base import WorkerHandler


# ---------------------------------------------------------------------------
# PythonHandler
# ---------------------------------------------------------------------------

class PythonHandler(WorkerHandler):
    """
    Persistent Python execution environment.

    Supports methods:
        execute(code)           — run Python code, return captured output
        get_state()             — return variable state (types, previews)
        save_state(path)        — serialize _globals to disk with dill
        load_state(path)        — deserialize _globals from disk with dill
        inject(code)            — execute code silently (setup / DB connections)
        reset_state()           — clear all user variables

    Required packages in the subprocess Python: dill
    """

    REQUIRED_DEPS = ["dill"]
    OPTIONAL_DEPS = [
        ("pandas",       "pandas"),
        ("numpy",        "numpy"),
        ("scipy",        "scipy"),
        ("statsmodels",  "statsmodels"),
        ("sklearn",      "scikit-learn"),
        ("matplotlib",   "matplotlib"),
        ("seaborn",      "seaborn"),
        ("tableone",     "tableone"),
        ("sqlalchemy",   "SQLAlchemy"),
        ("openpyxl",     "Excel read/write"),
    ]

    # Modules blocked when MEDDS_CODE_GATE=true
    _BLOCKED_MODULES = {
        'subprocess', 'socket', 'ftplib', 'smtplib', 'telnetlib',
        'xmlrpc', 'socketserver',
    }
    # (module, attribute) pairs blocked when MEDDS_CODE_GATE=true
    _BLOCKED_CALLS = {
        ('os', 'system'), ('os', 'popen'), ('os', 'execv'), ('os', 'execve'),
        ('os', 'execvp'), ('os', 'execvpe'), ('os', 'spawnv'), ('os', 'spawnve'),
        ('os', 'spawnvp'), ('os', 'spawnl'), ('os', 'spawnle'),
    }

    def __init__(self, work_dir: str, **kwargs):
        self.work_dir = os.path.abspath(work_dir)

        missing = []
        for dep in self.REQUIRED_DEPS:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        if missing:
            raise ImportError(
                f"Missing required packages in subprocess Python: {', '.join(missing)}. "
                f"Install them in the target Python environment ({sys.executable})."
            )

        self._available_libs: List[str] = []
        self._missing_optional: List[str] = []
        for import_name, display_name in self.OPTIONAL_DEPS:
            try:
                __import__(import_name)
                self._available_libs.append(display_name)
            except ImportError:
                self._missing_optional.append(display_name)

        self._globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "WORK_DIR":    self.work_dir,
            "UPLOADS_DIR": os.path.join(self.work_dir, "uploads"),
            "OUTPUTS_DIR": os.path.join(self.work_dir, "outputs"),
            "SCRIPTS_DIR": os.path.join(self.work_dir, "scripts"),
            "INTERNAL_DIR": os.path.join(self.work_dir, "internal"),
        }
        self._locals: Dict[str, Any] = {}
        self._setup_environment()

    # ------------------------------------------------------------------
    # WorkerHandler interface
    # ------------------------------------------------------------------

    def get_ready_info(self) -> Dict[str, Any]:
        return {
            "python_version":   sys.version.split(" ")[0],
            "available_libs":   self._available_libs,
            "missing_optional": self._missing_optional,
        }

    def dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "execute":
            output = self._execute(params.get("code", ""))
            return {"output": output}

        elif method == "inject":
            error = self._inject(params.get("code", ""))
            return {"error": error}

        elif method == "get_state":
            return {"variables": self._get_state()}

        elif method == "save_state":
            path = params.get("path")
            if not path:
                raise ValueError("save_state requires 'path'")
            self._save_state(path)
            return {}

        elif method == "load_state":
            path = params.get("path")
            if not path:
                raise ValueError("load_state requires 'path'")
            self._load_state(path)
            return {}

        elif method == "reset_state":
            self._reset_state()
            return {}

        else:
            raise ValueError(f"PythonHandler: unknown method '{method}'")

    def on_shutdown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Private: execution
    # ------------------------------------------------------------------

    def _setup_environment(self):
        setup_code = f'''
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

WORK_DIR = "{self.work_dir}"
UPLOADS_DIR  = os.path.join(WORK_DIR, "uploads")
OUTPUTS_DIR  = os.path.join(WORK_DIR, "outputs")
SCRIPTS_DIR  = os.path.join(WORK_DIR, "scripts")
INTERNAL_DIR = os.path.join(WORK_DIR, "internal")

os.chdir(WORK_DIR)
'''
        optional_imports = '''
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy
    import statsmodels.api as sm
except ImportError:
    pass
try:
    from tableone import TableOne
except ImportError:
    pass
'''
        self._inject(setup_code)
        self._inject(optional_imports)

    def _check_code_safety(self, tree: ast.AST) -> List[str]:
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split('.')[0]
                    if top in self._BLOCKED_MODULES:
                        issues.append(f"blocked import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split('.')[0]
                    if top in self._BLOCKED_MODULES:
                        issues.append(f"blocked import: from {node.module}")
            elif isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)):
                    pair = (node.func.value.id, node.func.attr)
                    if pair in self._BLOCKED_CALLS:
                        issues.append(f"blocked call: {node.func.value.id}.{node.func.attr}()")
        return issues

    def _execute(self, code: str) -> str:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_stdout = io.StringIO()
        sys.stderr = captured_stderr = io.StringIO()

        result = None
        error = None

        try:
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            else:
                if os.environ.get("MEDDS_CODE_GATE", "false").lower() in ("true", "1", "yes"):
                    issues = self._check_code_safety(tree)
                    if issues:
                        return "[Blocked] Code contains restricted operations: " + "; ".join(issues)

                statements = tree.body
                for i, stmt in enumerate(statements):
                    is_last = (i == len(statements) - 1)
                    try:
                        if is_last and isinstance(stmt, ast.Expr):
                            expr_code = compile(
                                ast.Expression(body=stmt.value), '<string>', 'eval'
                            )
                            result = eval(expr_code, self._globals, self._locals)
                        else:
                            stmt_code = compile(
                                ast.Module(body=[stmt], type_ignores=[]), '<string>', 'exec'
                            )
                            exec(stmt_code, self._globals, self._locals)
                        self._globals.update(self._locals)
                    except Exception as e:
                        error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                        break
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        output_parts = []
        stdout_content = captured_stdout.getvalue()
        stderr_content = captured_stderr.getvalue()

        if stdout_content:
            output_parts.append(stdout_content.rstrip())
        if stderr_content:
            output_parts.append(f"[stderr]\n{stderr_content.rstrip()}")
        if error:
            output_parts.append(f"[Error]\n{error}")
        elif result is not None:
            output_parts.append(self._format_result(result))

        return "\n".join(output_parts) if output_parts else "(No output)"

    def _inject(self, code: str) -> Optional[str]:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        error = None
        try:
            exec(code, self._globals, self._locals)
            self._globals.update(self._locals)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return error

    def _format_result(self, result: Any) -> str:
        if hasattr(result, 'to_string'):
            try:
                if hasattr(result, 'shape'):
                    rows = result.shape[0]
                    cols = result.shape[1] if len(result.shape) == 2 else 1
                    if rows > 50:
                        return (
                            f"DataFrame with {rows} rows x {cols} columns:\n"
                            f"{result.head(25).to_string()}\n...\n{result.tail(25).to_string()}"
                        )
                return result.to_string(max_rows=50)
            except Exception:
                pass
        result_str = repr(result)
        if len(result_str) > 4000:
            return f"{result_str[:4000]}...\n(output truncated, {len(result_str)} chars total)"
        return result_str

    # ------------------------------------------------------------------
    # Private: state management
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._globals = {
            "__builtins__": __builtins__,
            "WORK_DIR":    self.work_dir,
            "UPLOADS_DIR": os.path.join(self.work_dir, "uploads"),
            "OUTPUTS_DIR": os.path.join(self.work_dir, "outputs"),
            "SCRIPTS_DIR": os.path.join(self.work_dir, "scripts"),
            "INTERNAL_DIR": os.path.join(self.work_dir, "internal"),
        }
        self._locals = {}
        self._setup_environment()

    def _save_state(self, path: str):
        import dill
        state_to_save = {
            k: v for k, v in self._globals.items()
            if k not in ('db_engine', 'conn')
        }
        with open(path, 'wb') as f:
            dill.dump(state_to_save, f)

    def _load_state(self, path: str):
        import dill
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self._globals.update(dill.load(f))

    def _get_state(self) -> List[Dict[str, Any]]:
        excluded = {
            '__builtins__', 'WORK_DIR', 'UPLOADS_DIR', 'OUTPUTS_DIR',
            'SCRIPTS_DIR', 'INTERNAL_DIR', 'quit', 'exit', 'get_ipython'
        }
        state_data = []

        for k, v in self._globals.items():
            if k.startswith('_') or k in excluded:
                continue

            type_name = type(v).__name__
            module_name = type(v).__module__
            full_type = f"{module_name}.{type_name}" if module_name != "builtins" else type_name
            value_info = ""
            preview_content = ""
            is_error = False

            try:
                if "DataFrame" in type_name or "pandas" in full_type:
                    if hasattr(v, 'shape'):
                        value_info = f"({v.shape[0]}x{v.shape[1]})"
                    if hasattr(v, 'head') and hasattr(v, 'to_html'):
                        preview_content = v.head(500).to_html(classes='df-table', border=0, index=False)

                elif "Figure" in type_name and hasattr(v, 'savefig'):
                    buf = io.BytesIO()
                    v.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode('utf-8')
                    value_info = "Image"
                    preview_content = f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;">'

                elif "ndarray" in type_name:
                    if hasattr(v, 'shape'):
                        value_info = str(v.shape)
                    preview_content = str(v)

                elif "module" == type_name or "function" == type_name or "method" in type_name:
                    value_info = "Module/Func"
                    preview_content = str(v)

                else:
                    s = str(v)
                    value_info = s[:20] + "..." if len(s) > 20 else s
                    preview_content = s[:2000] + ("..." if len(s) > 2000 else "")

            except Exception as e:
                is_error = True
                value_info = "Error"
                preview_content = f"Could not serialize: {str(e)}"

            state_data.append({
                "name": k,
                "type": type_name,
                "value": value_info,
                "preview": preview_content,
                "is_error": is_error,
            })

        return state_data
