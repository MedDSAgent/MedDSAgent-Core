"""
WorkerHandler implementations that run inside subprocess workers.

Each class is launched by worker_entry.py via:
    python -m medds_agent.worker_entry medds_agent.worker_handlers.<ClassName> --work_dir /path

Available handlers:
    PythonHandler   — persistent Python execution environment (replaces in-process PythonExecutorTool logic)
    RHandler        — persistent R execution environment via rpy2 (replaces in-process RExecutorTool logic)
"""

import ast
import os
import sys
import io
import json
import base64
import traceback
from typing import Any, Dict, List, Optional

from medds_agent.workers import WorkerHandler


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

        # --- Check required dependencies ---
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

        # --- Check optional dependencies ---
        self._available_libs: List[str] = []
        self._missing_optional: List[str] = []
        for import_name, display_name in self.OPTIONAL_DEPS:
            try:
                __import__(import_name)
                self._available_libs.append(display_name)
            except ImportError:
                self._missing_optional.append(display_name)

        # --- Set up persistent namespace ---
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

    def on_shutdown(self):
        pass  # nothing to release

    # ------------------------------------------------------------------
    # Private: execution
    # ------------------------------------------------------------------

    def _setup_environment(self):
        """Pre-import common libraries into the execution namespace."""
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
        """Return list of issues if dangerous patterns are found in the AST."""
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
        """Execute code with full output capture. Returns formatted output string."""
        # Redirect stdout/stderr (real ones have been saved in worker_entry.py)
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
        """Execute code silently (for setup/DB injection). Returns error string or None."""
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
        """Return variable metadata for the UI, same format as the old PythonExecutorTool.get_state()."""
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


# ---------------------------------------------------------------------------
# RHandler
# ---------------------------------------------------------------------------

class RHandler(WorkerHandler):
    """
    Persistent R execution environment via rpy2.

    Supports methods:
        execute(code)           — run R code, return captured output
        get_state()             — return R environment variable metadata
        save_state(path)        — save environment to .RData file
        load_state(path)        — load .RData file into environment
        inject(code)            — execute R code silently (setup / DB connections)
        reset_state()           — clear the R environment

    Required packages in the subprocess Python: rpy2
    """

    REQUIRED_DEPS = ["rpy2"]

    # Regex patterns for dangerous R calls, checked when MEDDS_CODE_GATE=true
    _BLOCKED_R_PATTERNS = [
        (r'\bsystem\s*\(',        'system()'),
        (r'\bsystem2\s*\(',       'system2()'),
        (r'\bshell\s*\(',         'shell()'),
        (r'\bshell\.exec\s*\(',   'shell.exec()'),
        (r'\bdownload\.file\s*\(', 'download.file()'),
        (r'\bsocketConnection\s*\(', 'socketConnection()'),
        (r'\burl\s*\(',           'url()'),
    ]

    def __init__(self, work_dir: str, **kwargs):
        self.work_dir = os.path.abspath(work_dir)

        # --- Check required dependencies ---
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr
            import rpy2.rinterface_lib.callbacks as rcallbacks
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter

            self._robjects = robjects
            self._rcallbacks = rcallbacks
            self._pandas2ri = pandas2ri
            self._localconverter = localconverter
        except ImportError:
            raise ImportError(
                f"rpy2 is not installed in the subprocess Python ({sys.executable}). "
                "Install it to use the R executor."
            )

        # --- Set up persistent R environment ---
        self.env = self._robjects.Environment()
        self._setup_environment()

    # ------------------------------------------------------------------
    # WorkerHandler interface
    # ------------------------------------------------------------------

    def get_ready_info(self) -> Dict[str, Any]:
        try:
            r_version = str(self._robjects.r('R.version.string')[0])
        except Exception:
            r_version = "unknown"
        return {
            "r_version": r_version,
            "python_version": sys.version.split(" ")[0],
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
            raise ValueError(f"RHandler: unknown method '{method}'")

    def on_shutdown(self):
        pass

    # ------------------------------------------------------------------
    # Private: execution
    # ------------------------------------------------------------------

    def _setup_environment(self):
        self._inject(f'''
            WORK_DIR <- "{self.work_dir}"
            UPLOADS_DIR  <- file.path(WORK_DIR, "uploads")
            OUTPUTS_DIR  <- file.path(WORK_DIR, "outputs")
            SCRIPTS_DIR  <- file.path(WORK_DIR, "scripts")
            INTERNAL_DIR <- file.path(WORK_DIR, "internal")
            setwd(WORK_DIR)
        ''')

    def _check_r_code_safety(self, code: str) -> List[str]:
        import re
        issues = []
        for pattern, label in self._BLOCKED_R_PATTERNS:
            if re.search(pattern, code):
                issues.append(f"blocked call: {label}")
        return issues

    def _execute(self, code: str) -> str:
        if os.environ.get("MEDDS_CODE_GATE", "false").lower() in ("true", "1", "yes"):
            issues = self._check_r_code_safety(code)
            if issues:
                return "[Blocked] Code contains restricted operations: " + "; ".join(issues)

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        original_print  = self._rcallbacks.consolewrite_print
        original_warn   = self._rcallbacks.consolewrite_warnerror

        def _capture_print(x): stdout_buffer.write(x)
        def _capture_warn(x):  stderr_buffer.write(x)

        self._rcallbacks.consolewrite_print    = _capture_print
        self._rcallbacks.consolewrite_warnerror = _capture_warn

        result = None
        error  = None

        try:
            with self._localconverter(
                self._robjects.default_converter + self._pandas2ri.converter
            ):
                self._robjects.r(f'setwd("{self.work_dir}")')
                try:
                    parsed = self._robjects.r.parse(text=code)
                except Exception as e:
                    error = str(e)
                else:
                    for expr in parsed:
                        try:
                            result = self._robjects.r['eval'](expr, envir=self.env)
                        except Exception as e:
                            error = str(e)
                            break
        except Exception as e:
            error = str(e)
        finally:
            self._rcallbacks.consolewrite_print    = original_print
            self._rcallbacks.consolewrite_warnerror = original_warn

        output_parts = []
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()

        if stdout_content:
            output_parts.append(stdout_content.rstrip())
        if stderr_content:
            output_parts.append(f"[stderr]\n{stderr_content.rstrip()}")
        if error:
            output_parts.append(f"[Error]\n{error}")
        elif result is not None:
            try:
                with self._localconverter(
                    self._robjects.default_converter + self._pandas2ri.converter
                ):
                    capture_res = self._robjects.r['capture.output'](
                        self._robjects.r['print'](result)
                    )
                output_parts.append("\n".join(capture_res))
            except Exception:
                output_parts.append(str(result))

        return "\n".join(output_parts) if output_parts else "(No output)"

    def _inject(self, code: str) -> Optional[str]:
        try:
            with self._localconverter(
                self._robjects.default_converter + self._pandas2ri.converter
            ):
                self._robjects.r['eval'](
                    self._robjects.r.parse(text=code),
                    envir=self.env
                )
        except Exception as e:
            return str(e)
        return None

    # ------------------------------------------------------------------
    # Private: state management
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.env = self._robjects.Environment()
        self._setup_environment()

    def _save_state(self, path: str):
        with self._localconverter(
            self._robjects.default_converter + self._pandas2ri.converter
        ):
            env_keys = list(self.env.keys())
            if env_keys:
                self._robjects.r['save'](
                    list=self._robjects.StrVector(env_keys),
                    file=path,
                    envir=self.env
                )
            else:
                self._robjects.r['save'](list=self._robjects.StrVector([]), file=path)

    def _load_state(self, path: str):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with self._localconverter(
                self._robjects.default_converter + self._pandas2ri.converter
            ):
                self._robjects.r['load'](file=path, envir=self.env)

    _EXCLUDED_VARS = {'WORK_DIR', 'UPLOADS_DIR', 'OUTPUTS_DIR', 'SCRIPTS_DIR', 'INTERNAL_DIR'}

    def _r_check(self, func_name: str, obj) -> bool:
        try:
            return bool(self._robjects.r[func_name](obj)[0])
        except Exception:
            return False

    def _r_class_str(self, obj) -> str:
        try:
            return str(self._robjects.r['class'](obj)[0])
        except Exception:
            return "unknown"

    def _r_capture_preview(self, obj, max_chars: int = 2000) -> str:
        try:
            capture_res = self._robjects.r['capture.output'](self._robjects.r['print'](obj))
            return "\n".join(capture_res)[:max_chars]
        except Exception:
            return str(obj)[:max_chars]

    def _get_state(self) -> List[Dict[str, Any]]:
        state_data = []
        with self._localconverter(
            self._robjects.default_converter + self._pandas2ri.converter
        ):
            for name in list(self.env.keys()):
                if name.startswith('.') or name in self._EXCLUDED_VARS:
                    continue

                obj = self.env[name]
                r_class = self._r_class_str(obj)
                value_info = ""
                preview_content = ""
                is_error = False

                try:
                    if (self._r_check('is.ggplot', obj)
                            if self._robjects.r('exists("is.ggplot")')[0] else False):
                        r_class = "ggplot"
                        value_info = "Plot"
                        try:
                            plot_func = self._robjects.r("""
                            function(p) {
                                tmp <- tempfile(fileext = ".png")
                                png(tmp, width=600, height=400)
                                print(p)
                                dev.off()
                                f <- file(tmp, "rb")
                                data <- readBin(f, "raw", n = file.info(tmp)$size)
                                close(f)
                                unlink(tmp)
                                return(data)
                            }
                            """)
                            image_bytes = bytes(plot_func(obj))
                            b64 = base64.b64encode(image_bytes).decode('utf-8')
                            preview_content = f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;">'
                        except Exception as e:
                            preview_content = f"Could not render plot: {str(e)}"

                    elif self._r_check('is.data.frame', obj):
                        r_class = "data.frame"
                        rows = int(self._robjects.r['nrow'](obj)[0])
                        cols = int(self._robjects.r['ncol'](obj)[0])
                        value_info = f"({rows}x{cols})"
                        try:
                            head_df = self._robjects.r['head'](obj, n=500)
                            pd_df = self._robjects.conversion.get_conversion().rpy2py(head_df)
                            if hasattr(pd_df, 'to_html'):
                                preview_content = pd_df.to_html(classes='df-table', border=0, index=False)
                            else:
                                preview_content = self._r_capture_preview(obj)
                        except Exception:
                            preview_content = self._r_capture_preview(obj)

                    elif self._r_check('is.matrix', obj):
                        r_class = "matrix"
                        try:
                            dims = list(self._robjects.r['dim'](obj))
                            value_info = f"({int(dims[0])}x{int(dims[1])})"
                        except Exception:
                            pass
                        preview_content = self._r_capture_preview(obj)

                    elif self._r_check('is.list', obj):
                        r_class = "list"
                        try:
                            value_info = f"({int(self._robjects.r['length'](obj)[0])})"
                        except Exception:
                            pass
                        preview_content = self._r_capture_preview(obj)

                    elif self._r_check('is.function', obj):
                        r_class = "function"
                        preview_content = self._r_capture_preview(obj)

                    elif self._r_check('is.factor', obj):
                        r_class = "factor"
                        try:
                            length = int(self._robjects.r['length'](obj)[0])
                            n_levels = int(self._robjects.r['nlevels'](obj)[0])
                            value_info = f"({length}) [{n_levels} levels]"
                        except Exception:
                            pass
                        preview_content = self._r_capture_preview(obj)

                    elif self._r_check('is.atomic', obj):
                        if self._r_check('is.integer', obj):
                            r_class = "integer"
                        elif self._r_check('is.numeric', obj):
                            r_class = "numeric"
                        elif self._r_check('is.character', obj):
                            r_class = "character"
                        elif self._r_check('is.logical', obj):
                            r_class = "logical"
                        try:
                            length = int(self._robjects.r['length'](obj)[0])
                            value_info = f"({length})"
                            if length == 1:
                                val = str(self._robjects.r['as.character'](obj)[0])
                                value_info = val[:20] + "..." if len(val) > 20 else val
                        except Exception:
                            pass
                        preview_content = self._r_capture_preview(obj)

                    else:
                        value_info = r_class
                        preview_content = self._r_capture_preview(obj)

                except Exception as e:
                    is_error = True
                    value_info = "Error"
                    preview_content = str(e)

                state_data.append({
                    "name": name,
                    "type": r_class,
                    "value": value_info,
                    "preview": preview_content,
                    "is_error": is_error,
                })

        return state_data
