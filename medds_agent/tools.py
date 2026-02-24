"""
Tools for the Analyst Agent.

This module provides tools that the agent can use to interact with:
- Python execution
- R execution
- File system exploration
"""
import ast
import os
import sys
import io
import abc
import base64
import shutil
from typing import List, Dict, Any, Optional

try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rinterface_lib.callbacks as rcallbacks
    
    # Add these imports:
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False


class Tool(abc.ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abc.abstractmethod
    def execute(self, param: Dict) -> str:
        """Execute the tool with given parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_tool_call_schema(self) -> Dict:
        """Return OpenAI-compatible tool schema."""
        raise NotImplementedError

class PythonExecutorTool(Tool):
    """
    Full Python executor for local or Docker-isolated environments.

    Features:
    - Persistent state across executions (variables, imports, DataFrames)
    - Pre-imported data science libraries
    - Full file system access within WORK_DIR
    - Supports database connections (write your own connection code)
    """

    def __init__(self, work_dir: str):
        """
        Initialize the Python executor.

        Parameters:
        ----------
        work_dir : str
            The working directory for file operations.
        """
        self.work_dir = os.path.abspath(work_dir)
        self.python_version = sys.version.split(" ")[0]
        
        # Persistent namespace for state across executions
        self._globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "WORK_DIR": self.work_dir,
            "UPLOADS_DIR": os.path.join(self.work_dir, "uploads"),
            "OUTPUTS_DIR": os.path.join(self.work_dir, "outputs"),
            "INTERNAL_DIR": os.path.join(self.work_dir, "internal"),
            "SCRIPTS_DIR": os.path.join(self.work_dir, "scripts"),
        }
        self._locals: Dict[str, Any] = {}
        
        # Pre-import common data science libraries
        self._setup_environment()
        
        # Build description with available libraries
        self.description = self._build_description()
        super().__init__(name="PythonExecutor", description=self.description)
    
    def _setup_environment(self):
        """Pre-import common libraries into the execution namespace."""
        # Core setup - always runs
        setup_code = f'''
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

WORK_DIR = "{self.work_dir}"
UPLOADS_DIR = os.path.join(WORK_DIR, "uploads")
OUTPUTS_DIR = os.path.join(WORK_DIR, "outputs")
SCRIPTS_DIR = os.path.join(WORK_DIR, "scripts")
INTERNAL_DIR = os.path.join(WORK_DIR, "internal")
os.chdir(WORK_DIR)
'''
        exec(setup_code, self._globals, self._locals)
        self._globals.update(self._locals)
        
        # Optional imports - fail silently if not installed
        optional_imports = [
            # Data manipulation
            "import pandas as pd",
            "import numpy as np",
            # Statistical analysis
            "import scipy",
            "import statsmodels.api as statsmodels",
            "import sklearn",
            "import seaborn as sns",
            "import tableone",
            # Excel support
            "import openpyxl",
            # Display settings
            "pd.set_option('display.max_columns', 50)",
            "pd.set_option('display.width', None)",
            "pd.set_option('display.max_rows', 100)",
            # Database connections
            "from sqlalchemy import create_engine, text",
            "import oracledb",
            # Visualization
            "import matplotlib.pyplot as plt",
            "import matplotlib",
            "matplotlib.use('Agg')",  # Non-interactive backend for server
        ]
        
        for imp in optional_imports:
            try:
                exec(imp, self._globals, self._locals)
                self._globals.update(self._locals)
            except (ImportError, NameError, Exception):
                pass
    
    def _build_description(self) -> str:
        """Build a description including available libraries."""
        available_libs = []
        check_libs = [
            ('pandas', 'pd'), 
            ('numpy', 'np'), 
            ('tableone', 'tableone'),
            ('matplotlib', 'plt'),
            ('seaborn', 'sns'),
            ('scipy', 'scipy'),
            ('sklearn', 'sklearn'),
            ('statsmodels', 'statsmodels'),
            ('sqlalchemy', 'SQLAlchemy'),
            ('openpyxl', 'Excel read/write'),
        ]
        
        for lib, display in check_libs:
            try:
                __import__(lib)
                available_libs.append(display)
            except ImportError:
                pass
        
        libs_str = ", ".join(available_libs) if available_libs else "standard library"

        return (
            f"Executes Python {self.python_version} code with full permissions. "
            f"State persists across calls (variables, imports, DataFrames). "
            f"Use `print()` to output results. "
            f"Pre-imported: {libs_str}. "
            f"Directories: UPLOADS_DIR for user files, OUTPUTS_DIR for agent outputs, SCRIPTS_DIR for scripts shared with user, INTERNAL_DIR for internal use. "
            f"For database connections, use the pre-configured `db_engine` or `conn` object if available in state."
        )
    
    def execute(self, param: Dict) -> str:
        """
        Execute Python code with full permissions.
        
        Parameters:
        ----------
        param : Dict
            Must contain 'code' key with Python code string.
        
        Returns:
        -------
        str
            Captured stdout/stderr and the repr of the last expression if any.
        """
        if not isinstance(param, dict) or "code" not in param:
            raise ValueError("Parameter must be a dictionary with 'code' key.")
        
        code = param["code"]
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_stdout = io.StringIO()
        sys.stderr = captured_stderr = io.StringIO()
        
        result = None
        error = None

        try:
            # Parse the entire code block upfront to catch syntax errors and split
            # into individual top-level statements for statement-by-statement execution.
            # This ensures variables from successful statements are committed to the
            # environment even when a later statement raises an error.
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                import traceback
                error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            else:
                statements = tree.body
                for i, stmt in enumerate(statements):
                    is_last = (i == len(statements) - 1)
                    try:
                        if is_last and isinstance(stmt, ast.Expr):
                            # Last statement is a bare expression: evaluate it so
                            # its value is returned (mirrors the original eval() path).
                            expr_code = compile(
                                ast.Expression(body=stmt.value), '<string>', 'eval'
                            )
                            result = eval(expr_code, self._globals, self._locals)
                        else:
                            stmt_code = compile(
                                ast.Module(body=[stmt], type_ignores=[]), '<string>', 'exec'
                            )
                            exec(stmt_code, self._globals, self._locals)
                        # Persist state after every successful statement.
                        self._globals.update(self._locals)
                    except Exception as e:
                        import traceback
                        error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                        break  # Stop at first error; prior statements are already in env
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # Build output
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
    
    def _format_result(self, result: Any) -> str:
        """Format execution result for display."""
        # Handle DataFrames specially
        if hasattr(result, 'to_string'):
            try:
                # Limit output size
                if hasattr(result, 'shape'):
                    rows = result.shape[0]
                    cols = result.shape[1] if len(result.shape) == 2 else 1
                    if rows > 50:
                        return f"DataFrame with {rows} rows x {cols} columns:\n{result.head(25).to_string()}\n...\n{result.tail(25).to_string()}"
                return result.to_string(max_rows=50)
            except Exception:
                pass
        
        # Handle large collections
        result_str = repr(result)
        if len(result_str) > 4000:
            return f"{result_str[:4000]}...\n(output truncated, {len(result_str)} chars total)"
        
        return result_str

    def get_state(self) -> List[Dict[str, Any]]:
        """Get current variable state with metadata for UI."""
        # Filter out builtins and system variables
        excluded = {'__builtins__', 'WORK_DIR', 'UPLOADS_DIR', 'OUTPUTS_DIR', 'SCRIPTS_DIR', 'INTERNAL_DIR', 'quit', 'exit', 'get_ipython'}
        state_data = []
        
        for k, v in self._globals.items():
            if k.startswith('_') or k in excluded:
                continue
            
            # Determine basic type info
            type_name = type(v).__name__
            module_name = type(v).__module__
            full_type = f"{module_name}.{type_name}" if module_name != "builtins" else type_name
            
            value_info = ""
            preview_content = ""
            is_error = False

            try:
                # 1. Pandas DataFrame
                if "DataFrame" in type_name or "pandas" in full_type:
                    if hasattr(v, 'shape'):
                        value_info = f"({v.shape[0]}x{v.shape[1]})"
                    if hasattr(v, 'head') and hasattr(v, 'to_html'):
                        # Render a nice Bootstrap table
                        preview_content = v.head(500).to_html(classes='df-table', border=0, index=False)
                
                # 2. Matplotlib Figure
                elif "Figure" in type_name and hasattr(v, 'savefig'):
                    # Convert to Base64 image
                    buf = io.BytesIO()
                    v.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode('utf-8')
                    value_info = "Image"
                    preview_content = f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;">'
                
                # 3. Numpy Array
                elif "ndarray" in type_name:
                    if hasattr(v, 'shape'):
                        value_info = str(v.shape)
                    preview_content = str(v)

                # 4. Modules/Functions (for filtering)
                elif "module" == type_name or "function" == type_name or "method" in type_name:
                    value_info = "Module/Func"
                    preview_content = str(v)

                # 5. Standard Primitives
                else:
                    s = str(v)
                    # Truncate for the "value" badge
                    value_info = s[:20] + "..." if len(s) > 20 else s
                    # Preview gets more content but still safe
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
                "is_error": is_error
            })
            
        return state_data
    
    def reset_state(self):
        """Reset execution state to initial."""
        self._globals = {
            "__builtins__": __builtins__,
            "WORK_DIR": self.work_dir,
            "UPLOADS_DIR": os.path.join(self.work_dir, "uploads"),
            "OUTPUTS_DIR": os.path.join(self.work_dir, "outputs"),
            "SCRIPTS_DIR": os.path.join(self.work_dir, "scripts"),
            "INTERNAL_DIR": os.path.join(self.work_dir, "internal"),
        }
        self._locals = {}
        self._setup_environment()

    def get_tool_call_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute.",
                        },
                    },
                    "required": ["code"],
                },
            },
        }

class RExecutorTool(Tool):
    """
    R Executor using rpy2.
    """
    def __init__(self, work_dir: str):
        if not HAS_RPY2:
            raise ImportError("rpy2 is not installed. Cannot use RExecutorTool.")
            
        self.work_dir = os.path.abspath(work_dir)
        self.name = "RExecutor"
        
        # Initialize R environment
        # We use a specific environment to isolate variables as much as possible,
        # although libraries are loaded globally in the embedded R process.
        self.env = robjects.Environment()
        
        self.description = (
            "Executes R code. State persists across calls within the session. "
            "Use standard R syntax. "
            f"Directories: UPLOADS_DIR for user files, OUTPUTS_DIR for agent outputs, SCRIPTS_DIR for scripts shared with user, INTERNAL_DIR for internal use. "
        )
        super().__init__(name=self.name, description=self.description)
        
        self._setup_environment()

    def _setup_environment(self):
        """Initialize the R environment variables."""
        # Set directory variables in R
        self.execute({"code": f"""
            WORK_DIR <- "{self.work_dir}"
            UPLOADS_DIR <- file.path(WORK_DIR, "uploads")
            OUTPUTS_DIR <- file.path(WORK_DIR, "outputs")
            SCRIPTS_DIR <- file.path(WORK_DIR, "scripts")
            INTERNAL_DIR <- file.path(WORK_DIR, "internal")
            setwd(WORK_DIR)
        """})

    def execute(self, param: Dict) -> str:
        """Execute R code."""
        if not isinstance(param, dict) or "code" not in param:
            raise ValueError("Parameter must be a dictionary with 'code' key.")

        code = param["code"]

        # Capture stdout/stderr
        # We define a custom callback to write to our string buffer
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        def write_console_ex(output, otype):
            if otype == 0: # Output
                stdout_buffer.write(output)
            else: # Message/Error
                stderr_buffer.write(output)

        # Hijack rpy2 callbacks
        # Note: rpy2 callbacks are global. In a multi-threaded server this can leak output
        # to other threads if they run concurrently.
        # PythonExecutor has similar issues with os.chdir.
        original_console = rcallbacks.consolewrite_print

        # Only simple write_console is exposed easily in rinterface_lib
        # We wrap the python function to match signature
        def print_capture(x):
            stdout_buffer.write(x)

        def warn_capture(x):
            stderr_buffer.write(x)

        rcallbacks.consolewrite_print = print_capture
        rcallbacks.consolewrite_warnerror = warn_capture

        result = None
        error = None

        try:
            # Use localconverter to ensure rpy2 conversion rules are available
            # in worker threads (asyncio.to_thread). The rules live in a
            # contextvars.ContextVar that may not propagate automatically.
            with localconverter(robjects.default_converter + pandas2ri.converter):
                # Ensure working directory is set (race condition mitigation)
                robjects.r(f'setwd("{self.work_dir}")')

                # Parse all top-level expressions upfront to catch syntax errors.
                # Evaluating expression-by-expression ensures variables from
                # successful statements are committed to the R environment even
                # when a later statement raises an error.
                try:
                    parsed = robjects.r.parse(text=code)
                except Exception as e:
                    error = str(e)
                else:
                    for expr in parsed:
                        try:
                            result = robjects.r['eval'](expr, envir=self.env)
                        except Exception as e:
                            error = str(e)
                            break  # Stop at first error; prior expressions are already in env

        except Exception as e:
            error = str(e)
        finally:
            # Restore callbacks
            rcallbacks.consolewrite_print = original_console
            rcallbacks.consolewrite_warnerror = original_console  # Restore default behavior

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
            # Format R result
            try:
                # Use R's print to string
                # We can use capture.output in R
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    capture_res = robjects.r['capture.output'](robjects.r['print'](result))
                res_str = "\n".join(capture_res)
                output_parts.append(res_str)
            except Exception:
                output_parts.append(str(result))

        return "\n".join(output_parts) if output_parts else "(No output)"

    # Variables to exclude from get_state (directory paths set during _setup_environment)
    _EXCLUDED_VARS = {'WORK_DIR', 'UPLOADS_DIR', 'OUTPUTS_DIR', 'SCRIPTS_DIR', 'INTERNAL_DIR'}

    def _r_check(self, func_name: str, obj) -> bool:
        """Call an R predicate function (e.g. is.data.frame) and return Python bool."""
        try:
            return bool(robjects.r[func_name](obj)[0])
        except:
            return False

    def _r_class_str(self, obj) -> str:
        """Get the primary R class string for an object."""
        try:
            cls_vec = robjects.r['class'](obj)
            return str(cls_vec[0])
        except:
            return "unknown"

    def _r_capture_preview(self, obj, max_chars: int = 2000) -> str:
        """Use capture.output(print(obj)) to get a text preview."""
        try:
            capture_res = robjects.r['capture.output'](robjects.r['print'](obj))
            raw = "\n".join(capture_res)
            return raw[:max_chars]
        except:
            return str(obj)[:max_chars]

    def get_state(self) -> List[Dict[str, Any]]:
        """Get R environment state with rich media support."""
        state_data = []

        # Use localconverter throughout to ensure conversion rules are available
        # in any thread context (asyncio.to_thread workers, etc.)
        with localconverter(robjects.default_converter + pandas2ri.converter):
            # List objects in the environment
            obj_names = list(self.env.keys())

            for name in obj_names:
                if name.startswith('.') or name in self._EXCLUDED_VARS:
                    continue

                obj = self.env[name]
                r_class = self._r_class_str(obj)
                value_info = ""
                preview_content = ""
                is_error = False

                try:
                    # --- 1. Handle ggplot/lattice objects (Figures) ---
                    if self._r_check('is.ggplot', obj) if robjects.r('exists("is.ggplot")')[0] else False:
                        r_class = "ggplot"
                        value_info = "Plot"
                        try:
                            plot_func = robjects.r("""
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
                            image_bytes_r = plot_func(obj)
                            image_bytes = bytes(image_bytes_r)
                            b64 = base64.b64encode(image_bytes).decode('utf-8')
                            preview_content = f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;">'
                        except Exception as e:
                            preview_content = f"Could not render plot: {str(e)}"

                    # --- 2. Handle DataFrames (data.frame, tibble, data.table) ---
                    elif self._r_check('is.data.frame', obj):
                        r_class = "data.frame"
                        rows = int(robjects.r['nrow'](obj)[0])
                        cols = int(robjects.r['ncol'](obj)[0])
                        value_info = f"({rows}x{cols})"
                        try:
                            head_df = robjects.r['head'](obj, n=500)
                            pd_df = robjects.conversion.get_conversion().rpy2py(head_df)
                            if hasattr(pd_df, 'to_html'):
                                preview_content = pd_df.to_html(classes='df-table', border=0, index=False)
                            else:
                                preview_content = self._r_capture_preview(obj)
                        except Exception as e:
                            preview_content = self._r_capture_preview(obj)

                    # --- 3. Handle Matrices ---
                    elif self._r_check('is.matrix', obj):
                        r_class = "matrix"
                        try:
                            dims = list(robjects.r['dim'](obj))
                            value_info = f"({int(dims[0])}x{int(dims[1])})"
                        except:
                            value_info = ""
                        preview_content = self._r_capture_preview(obj)

                    # --- 4. Handle Arrays ---
                    elif self._r_check('is.array', obj):
                        r_class = "array"
                        try:
                            dims = [int(d) for d in robjects.r['dim'](obj)]
                            value_info = f"({'x'.join(map(str, dims))})"
                        except:
                            value_info = ""
                        preview_content = self._r_capture_preview(obj)

                    # --- 5. Handle Lists (non-data.frame) ---
                    elif self._r_check('is.list', obj):
                        r_class = "list"
                        try:
                            length = int(robjects.r['length'](obj)[0])
                            value_info = f"({length})"
                        except:
                            pass
                        preview_content = self._r_capture_preview(obj)

                    # --- 6. Handle Functions ---
                    elif self._r_check('is.function', obj):
                        r_class = "function"
                        value_info = ""
                        preview_content = self._r_capture_preview(obj)

                    # --- 7. Handle Factors ---
                    elif self._r_check('is.factor', obj):
                        r_class = "factor"
                        try:
                            length = int(robjects.r['length'](obj)[0])
                            n_levels = int(robjects.r['nlevels'](obj)[0])
                            value_info = f"({length}) [{n_levels} levels]"
                        except:
                            pass
                        preview_content = self._r_capture_preview(obj)

                    # --- 8. Handle Vectors (numeric, integer, character, logical) ---
                    elif self._r_check('is.atomic', obj):
                        # Determine specific atomic type
                        if self._r_check('is.numeric', obj):
                            r_class = "integer" if self._r_check('is.integer', obj) else "numeric"
                        elif self._r_check('is.character', obj):
                            r_class = "character"
                        elif self._r_check('is.logical', obj):
                            r_class = "logical"
                        else:
                            r_class = self._r_class_str(obj)

                        try:
                            length = int(robjects.r['length'](obj)[0])
                            value_info = f"({length})"
                        except:
                            pass

                        # For short vectors, show the actual value
                        try:
                            length = int(robjects.r['length'](obj)[0])
                            if length == 1:
                                val = str(robjects.r['as.character'](obj)[0])
                                value_info = val[:20] + "..." if len(val) > 20 else val
                        except:
                            pass

                        preview_content = self._r_capture_preview(obj)

                    # --- 9. Default Fallback ---
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
                    "is_error": is_error
                })

        return state_data

    def save_state(self, path: str):
        """Save the R environment to a .RData file."""
        with localconverter(robjects.default_converter + pandas2ri.converter):
            env_keys = list(self.env.keys())
            if env_keys:
                robjects.r['save'](list=robjects.StrVector(env_keys), file=path, envir=self.env)
            else:
                robjects.r['save'](list=robjects.StrVector([]), file=path)

    def load_state(self, path: str):
        """Load an .RData file into the environment."""
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with localconverter(robjects.default_converter + pandas2ri.converter):
                robjects.r['load'](file=path, envir=self.env)
    
    def get_tool_call_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "R code to execute.",
                        },
                    },
                    "required": ["code"],
                },
            },
        }

class DocumentSearchTool(Tool):
    """
    Tool for browsing and reading parsed document sections.

    Uploaded documents (PDF, Word, etc.) are automatically parsed into
    hierarchical sections on upload. This tool lets the agent explore
    the structure and read specific sections without loading the full
    document into context.
    """

    def __init__(self, session_id: str, db):
        """
        Parameters
        ----------
        session_id : str
            The current session ID for scoping queries.
        db : InternalDatabase
            Reference to the shared database instance.
        """
        self.session_id = session_id
        self.db = db
        self.description = (
            "Browse and read parsed document sections. "
            "Uploaded documents (PDF, DOCX, PPTX, XLSX, Markdown, TXT) are automatically parsed into hierarchical sections on upload. "
            "Use this tool to explore document structure and read specific sections without loading the entire file. "
            "IMPORTANT: Always use this tool to read uploaded documents instead of FileSystem's read action. "
            "Actions: "
            "'list_documents' - list all parsed documents; "
            "'get_outline' - get the hierarchical section outline of a document; "
            "'read_section' - read the content of a specific section (includes child sections)."
        )
        super().__init__(name="DocumentSearch", description=self.description)

    def execute(self, param: Dict) -> str:
        if not param:
            param = {}

        action = param.get("action", "list_documents")

        try:
            if action == "list_documents":
                return self._list_documents()
            elif action == "get_outline":
                return self._get_outline(param.get("file"))
            elif action == "read_section":
                return self._read_section(param.get("file"), param.get("section_id"))
            else:
                return f"Error: Unknown action '{action}'. Use 'list_documents', 'get_outline', or 'read_section'."
        except Exception as e:
            return f"Error: {str(e)}"

    def _list_documents(self) -> str:
        docs = self.db.list_parsed_documents(self.session_id)
        if not docs:
            return "No parsed documents found. Upload a document (PDF, DOCX, PPTX, XLSX, MD, TXT) to get started."

        lines = ["Parsed documents:"]
        for doc in docs:
            section_count = len(self.db.get_document_sections(doc['document_id']))
            lines.append(f"  - {doc['file_name']} ({section_count} sections, parsed at {doc['parsed_at']})")
        return "\n".join(lines)

    def _get_outline(self, file_name: Optional[str]) -> str:
        if not file_name:
            return "Error: 'file' parameter is required for get_outline."

        doc_record = self.db.get_parsed_document(self.session_id, file_name)
        if not doc_record:
            return f"Error: No parsed document found for '{file_name}'. Check list_documents for available files."

        sections = self.db.get_document_sections(doc_record['document_id'])
        if not sections:
            return f"No sections found in '{file_name}'."

        lines = [f"Outline of '{file_name}':"]
        for s in sections:
            indent = "  " * (s['level'])
            lines.append(f"{indent}[{s['section_id']}] {s['title']}  ({len(s['content'])} chars)")
        return "\n".join(lines)

    def _read_section(self, file_name: Optional[str], section_id: Optional[str]) -> str:
        if not file_name:
            return "Error: 'file' parameter is required for read_section."
        if not section_id:
            return "Error: 'section_id' parameter is required for read_section."

        doc_record = self.db.get_parsed_document(self.session_id, file_name)
        if not doc_record:
            return f"Error: No parsed document found for '{file_name}'."

        document_id = doc_record['document_id']
        section = self.db.get_section_by_id(document_id, section_id)
        if not section:
            return f"Error: Section '{section_id}' not found in '{file_name}'. Use get_outline to see available sections."

        # Build output: this section + all descendant sections
        parts = [f"## [{section['section_id']}] {section['title']}\n"]
        if section['content']:
            parts.append(section['content'])

        # Recursively collect children
        self._collect_children(document_id, section_id, parts, depth=1)

        return "\n\n".join(parts)

    def _collect_children(self, document_id: int, parent_id: str, parts: List[str], depth: int):
        """Recursively collect child section content."""
        children = self.db.get_child_sections(document_id, parent_id)
        for child in children:
            header_prefix = "#" * min(depth + 2, 6)
            parts.append(f"{header_prefix} [{child['section_id']}] {child['title']}")
            if child['content']:
                parts.append(child['content'])
            self._collect_children(document_id, child['section_id'], parts, depth + 1)

    def get_tool_call_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["list_documents", "get_outline", "read_section"],
                            "description": (
                                "Action to perform: "
                                "'list_documents' to see all parsed documents, "
                                "'get_outline' to see the hierarchical section structure of a document, "
                                "'read_section' to read the full content of a specific section."
                            ),
                        },
                        "file": {
                            "type": "string",
                            "description": "The file name (e.g., 'data_dictionary.pdf'). Required for get_outline and read_section.",
                        },
                        "section_id": {
                            "type": "string",
                            "description": "The section ID to read (e.g., '1', '1.2', '2.3.1'). Required for read_section. Use get_outline to see available section IDs.",
                        },
                    },
                    "required": ["action"],
                },
            },
        }


class FileSystemTool(Tool):
    """
    Restricted File System tool.
    Allows listing, reading, writing, and deleting files in 'scripts', 'uploads', 'outputs', and 'internal' directories.
    """
    
    def __init__(self, dir: str, max_read_size: int = 1_048_576):
        self.dir = os.path.abspath(dir)
        self.allowed_subdirs = ["scripts", "uploads", "outputs", "internal"]
        self.max_read_size = max_read_size
        self.description = (
            "Interact with the file system. You can list, read, write, and delete files "
            "within the 'scripts/', 'uploads/', 'outputs/', and 'internal/' directories. "
            "IMPORTANT: Do NOT use this tool to read uploaded documents (PDF, DOCX, PPTX, XLSX, MD, TXT). "
            "Use the DocumentSearch tool instead, which provides parsed and indexed access to document sections. "
            "This tool is intended for code files, scripts, and small text files only."
        )
        super().__init__(name="FileSystem", description=self.description)

    def _validate_path(self, relative_path: str) -> str:
        """
        Validate that the path targets an allowed subdirectory.
        """
        clean_path = os.path.normpath(relative_path)
        if clean_path == "." or clean_path == "":
            return "" 

        is_allowed = any(clean_path.startswith(d) for d in self.allowed_subdirs)
        
        if not is_allowed:
            raise PermissionError(
                f"Access Denied: You are restricted to {self.allowed_subdirs}. "
                f"Cannot access '{clean_path}'."
            )
            
        target_path = os.path.abspath(os.path.join(self.dir, clean_path))
        
        if not target_path.startswith(self.dir):
            raise PermissionError("Access denied: Cannot traverse outside workspace.")
            
        return target_path

    def execute(self, param: Dict = None) -> str:
        """
        Execute file system actions: list, read, write, or delete.
        """
        if not param:
            param = {}
            
        action = param.get("action", "list")
        raw_path = param.get("path", "")
        
        try:
            if action == "list":
                if not raw_path or raw_path.strip() in [".", "/", ""]:
                    entries = []
                    for subdir in self.allowed_subdirs:
                        full_path = os.path.join(self.dir, subdir)
                        if os.path.exists(full_path):
                            count = len(os.listdir(full_path))
                            entries.append(f"[DIR]  {subdir}/ ({count} items)")
                    return "Contents of Workspace:\n" + "\n".join(entries)

                target_path = self._validate_path(raw_path)
                
                if not os.path.exists(target_path):
                    return f"Path does not exist: {raw_path}"
                
                if not os.path.isdir(target_path):
                    size = os.path.getsize(target_path)
                    return f"File: {os.path.basename(target_path)} ({self._format_size(size)})"
                
                entries = []
                for entry in sorted(os.listdir(target_path)):
                    if entry.startswith('.'): continue
                    
                    full_path = os.path.join(target_path, entry)
                    if os.path.isdir(full_path):
                        count = len(os.listdir(full_path))
                        entries.append(f"[DIR]  {entry}/ ({count} items)")
                    else:
                        size = os.path.getsize(full_path)
                        entries.append(f"[FILE] {entry} ({self._format_size(size)})")
                
                rel_path = os.path.relpath(target_path, self.dir)
                return f"Contents of {rel_path}:\n" + "\n".join(entries)

            elif action == "read":
                if not raw_path:
                    return "Error: 'path' is required for read action."
                
                target_path = self._validate_path(raw_path)
                
                if not os.path.exists(target_path):
                    return f"Error: File does not exist: {raw_path}"
                
                if os.path.isdir(target_path):
                    return f"Error: '{raw_path}' is a directory. Use 'list' instead."

                # File size check
                file_size = os.path.getsize(target_path)
                if file_size > self.max_read_size:
                    return (
                        f"Error: File '{raw_path}' is {self._format_size(file_size)}, "
                        f"which exceeds the {self._format_size(self.max_read_size)} limit. "
                        "For uploaded documents (PDF, DOCX, PPTX, XLSX, MD, TXT), use the DocumentSearch tool "
                        "to browse and read sections. For data files, use PythonExecutor to process in chunks."
                    )
                
                with open(target_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content

            elif action == "write":
                if not raw_path:
                    return "Error: 'path' is required for write action."
                
                content = param.get("content", "")
                target_path = self._validate_path(raw_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return f"Successfully wrote {len(content)} characters to {raw_path}"

            elif action == "delete":
                if not raw_path:
                    return "Error: 'path' is required for delete action."
                
                target_path = self._validate_path(raw_path)
                
                if not os.path.exists(target_path):
                    return f"Error: Path does not exist: {raw_path}"
                
                if os.path.isdir(target_path):
                    shutil.rmtree(target_path)
                    return f"Successfully deleted directory: {raw_path}"
                else:
                    os.remove(target_path)
                    return f"Successfully deleted file: {raw_path}"

            else:
                return f"Error: Unknown action '{action}'"
                
        except PermissionError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}" if unit != 'B' else f"{size} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def get_tool_call_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["list", "read", "write", "delete"],
                            "description": "Action to perform: list directory contents, read file content, write content to a file, or delete a file/directory.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path (e.g., 'uploads/data.csv' or 'outputs/report.md').",
                        },
                        "content": {
                            "type": "string",
                            "description": "String content to write (required for 'write' action).",
                        },
                    },
                    "required": ["action"],
                },
            },
        }