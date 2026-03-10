"""
Tools for the Analyst Agent.

This module provides tools that the agent can use to interact with:
- Python execution (AsyncTool — runs in subprocess via PythonHandler)
- R execution (AsyncTool — runs in subprocess via RHandler)
- File system exploration (sync)
- Document search (sync)
- Job management: job_wait, job_cancel (sync)
"""
import os
import sys
import abc
import shutil
from typing import List, Dict, Any, Optional

from medds_agent.job_manager import JobManager, ToolBusyError


class Tool(abc.ABC):
    """Abstract base class for all tools."""

    #: Override in subclasses. "sync" tools execute in-process immediately.
    #: "async" tools submit work to a subprocess worker and return a job_id.
    tool_type: str = "sync"

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

    def get_title(self, args: Dict) -> str:
        """
        Return a one-line display title for this tool call.

        The default implementation reads the LLM-provided 'title' key from args
        (used by code executor tools). Tools whose title can be derived from
        structured parameters should override this method instead.
        """
        return args.get("title", "")


class AsyncTool(Tool):
    """
    Base class for tools that run in a subprocess worker.

    Instead of executing in-process, AsyncTool.submit() sends work to a
    CodeWorker subprocess and returns a job_id immediately. The agent loop
    auto-waits for completion via JobManager.

    Subclasses must implement:
        submit(param) -> str          (returns job_id)
        get_tool_call_schema() -> dict
    """

    tool_type: str = "async"

    def __init__(self, name: str, description: str, job_manager):
        """
        Parameters
        ----------
        job_manager : JobManager
            The session's JobManager instance.
        """
        super().__init__(name, description)
        self.job_manager = job_manager

    def execute(self, param: Dict) -> str:
        # AsyncTools are never called via execute() in the agent loop.
        # The loop checks tool_type == "async" and calls submit() instead.
        raise RuntimeError(
            f"AsyncTool '{self.name}' does not support execute(). "
            "The agent loop should call submit() for async tools."
        )

    @abc.abstractmethod
    def submit(self, param: Dict) -> str:
        """
        Submit work to the subprocess worker.

        Parameters
        ----------
        param : dict
            Tool call arguments from the LLM.

        Returns
        -------
        str
            job_id — an opaque identifier the agent can use with job_wait / job_cancel.

        Raises
        ------
        ToolBusyError
            If the worker is already busy with another job.
        """
        raise NotImplementedError

class PythonExecutorTool(AsyncTool):
    """
    Python code executor — runs in a persistent subprocess worker (PythonHandler).

    Execution logic lives in worker_handlers.PythonHandler.
    This class submits code to the worker via JobManager and returns a job_id.
    The agent loop auto-waits for completion (see agents.py).
    """

    def __init__(self, job_manager: JobManager, ready_info: Optional[Dict] = None):
        """
        Parameters
        ----------
        job_manager : JobManager
            The session's JobManager, backed by a PythonHandler worker.
        ready_info : dict, optional
            The startup handshake payload from CodeWorker (available_libs, python_version).
            Used to build the tool description.
        """
        ready_info = ready_info or {}
        python_version = ready_info.get("python_version", sys.version.split(" ")[0])
        available_libs = ready_info.get("available_libs", [])
        libs_str = ", ".join(available_libs) if available_libs else "standard library"

        description = (
            f"Executes Python {python_version} code with full permissions. "
            f"State persists across calls (variables, imports, DataFrames). "
            f"Use `print()` to output results. "
            f"Pre-imported: {libs_str}. "
            f"Directories: UPLOADS_DIR for user files, OUTPUTS_DIR for agent outputs, "
            f"SCRIPTS_DIR for scripts shared with user, INTERNAL_DIR for internal use. "
            f"For database connections, use the pre-configured `db_engine` or `conn` object if available in state."
        )
        super().__init__(name="PythonExecutor", description=description, job_manager=job_manager)

    def submit(self, param: Dict) -> str:
        if not isinstance(param, dict) or "code" not in param:
            raise ValueError("Parameter must be a dictionary with 'code' key.")
        return self.job_manager.submit(
            tool_name=self.name,
            method="execute",
            params={"code": param["code"]},
        )

    def get_state(self) -> List[Dict[str, Any]]:
        """Fetch variable state from the worker subprocess (for UI env panel)."""
        data = self.job_manager.worker.send_command_sync("get_state", {})
        return data.get("variables", [])

    def get_tool_call_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": (
                                "A one-line summary of what this code does, shown in the UI when the code block is collapsed. "
                                "Be specific and concise (e.g., 'Load data from uploads/data.csv and inspect shape', "
                                "'Fit logistic regression for mortality with predictors age, sex, and BMI', "
                                "'Plot Kaplan-Meier survival curve by treatment group')."
                            ),
                        },
                        "code": {
                            "type": "string",
                            "description": "Python code to execute.",
                        },
                    },
                    "required": ["title", "code"],
                },
            },
        }


class RExecutorTool(AsyncTool):
    """
    R code executor — runs in a persistent subprocess worker (RHandler).

    Execution logic lives in worker_handlers.RHandler.
    This class submits code to the worker via JobManager and returns a job_id.
    The agent loop auto-waits for completion (see agents.py).
    """

    def __init__(self, job_manager: JobManager, ready_info: Optional[Dict] = None):
        """
        Parameters
        ----------
        job_manager : JobManager
            The session's JobManager, backed by an RHandler worker.
        ready_info : dict, optional
            The startup handshake payload from CodeWorker (r_version, etc.).
        """
        ready_info = ready_info or {}
        r_version = ready_info.get("r_version", "R")

        description = (
            f"Executes {r_version} code. State persists across calls within the session. "
            f"Use standard R syntax. "
            f"Directories: UPLOADS_DIR for user files, OUTPUTS_DIR for agent outputs, "
            f"SCRIPTS_DIR for scripts shared with user, INTERNAL_DIR for internal use."
        )
        super().__init__(name="RExecutor", description=description, job_manager=job_manager)

    def submit(self, param: Dict) -> str:
        if not isinstance(param, dict) or "code" not in param:
            raise ValueError("Parameter must be a dictionary with 'code' key.")
        return self.job_manager.submit(
            tool_name=self.name,
            method="execute",
            params={"code": param["code"]},
        )

    def get_state(self) -> List[Dict[str, Any]]:
        """Fetch variable state from the worker subprocess (for UI env panel)."""
        data = self.job_manager.worker.send_command_sync("get_state", {})
        return data.get("variables", [])

    def get_tool_call_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": (
                                "A one-line summary of what this code does, shown in the UI when the code block is collapsed. "
                                "Be specific and concise (e.g., 'Load data from uploads/data.csv and inspect shape', "
                                "'Fit logistic regression for mortality with predictors age, sex, and BMI', "
                                "'Plot Kaplan-Meier survival curve by treatment group')."
                            ),
                        },
                        "code": {
                            "type": "string",
                            "description": "R code to execute.",
                        },
                    },
                    "required": ["title", "code"],
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

    def get_title(self, args: Dict) -> str:
        action = args.get("action", "")
        file = args.get("file", "")
        section_id = args.get("section_id", "")
        if action == "list_documents":
            return "List documents"
        elif action == "get_outline":
            return f"Get outline of {file}" if file else "Get outline"
        elif action == "read_section":
            parts = "read section"
            if section_id:
                parts += f" {section_id}"
            if file:
                parts += f" of {file}"
            return parts.capitalize()
        return action

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

    def get_title(self, args: Dict) -> str:
        action = args.get("action", "")
        path = args.get("path", "")
        if action == "list":
            return f"List {path}" if path else "List workspace"
        elif action == "read":
            return f"Read file {path}" if path else "Read file"
        elif action == "write":
            return f"Write to {path}" if path else "Write file"
        elif action == "delete":
            return f"Delete {path}" if path else "Delete file"
        return action

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


class JobWaitTool(Tool):
    """
    Wait for a running async job to complete and collect its output.

    Blocks (in a thread) until the job finishes or max_sec is reached.
    Pass max_sec=0 for an instant status check with no waiting.

    Automatically injected into every agent that has at least one AsyncTool.
    """

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        super().__init__(
            name="job_wait",
            description=(
                "Wait for a running background job to complete and return its output. "
                "Use this after receiving a job_id from PythonExecutor or RExecutor "
                "when the job did not finish within the auto-wait window. "
                "Pass max_sec=0 to instantly check the current status without waiting."
            ),
        )

    def execute(self, param: Dict) -> str:
        job_id  = param.get("job_id", "").strip()
        max_sec = float(param.get("max_sec", 60))

        if not job_id:
            return "Error: job_id is required."

        job = self.job_manager.wait_sync(job_id, max_sec)

        from medds_agent.job_manager import (
            STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED, STATUS_TIMED_OUT
        )

        if job.status == STATUS_COMPLETED:
            return job.result or "(No output)"
        elif job.status == STATUS_FAILED:
            return f"[Job failed]\n{job.error}"
        elif job.status == STATUS_CANCELLED:
            return f"Job '{job_id}' was cancelled."
        elif job.status == STATUS_TIMED_OUT:
            from datetime import datetime
            elapsed = (datetime.now() - job.submitted_at).total_seconds()
            return (
                f"Job '{job_id}' is still running (elapsed: {elapsed:.1f}s, waited: {max_sec}s). "
                f"Call job_wait again with a longer max_sec, or job_cancel to abort."
            )
        else:
            return f"Job '{job_id}' status: {job.status}."

    def get_title(self, args: Dict) -> str:
        job_id = args.get("job_id", "")
        return f"Wait for job {job_id}" if job_id else "Wait for job"

    def get_tool_call_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The job_id returned by PythonExecutor or RExecutor.",
                        },
                        "max_sec": {
                            "type": "number",
                            "description": (
                                "Maximum seconds to wait for the job to complete. "
                                "Use 0 for an instant status check. Default: 60."
                            ),
                        },
                    },
                    "required": ["job_id"],
                },
            },
        }


class JobCancelTool(Tool):
    """
    Cancel a running async job.

    Kills the worker subprocess and restarts it cleanly.
    State accumulated by the cancelled job is lost.

    Automatically injected into every agent that has at least one AsyncTool.
    """

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        super().__init__(
            name="job_cancel",
            description=(
                "Cancel a running background job. "
                "The worker process is restarted — variables defined in the cancelled "
                "execution will not be available. Use this when a job is taking too long "
                "or you want to abandon the current execution."
            ),
        )

    def execute(self, param: Dict) -> str:
        job_id = param.get("job_id", "").strip()
        if not job_id:
            return "Error: job_id is required."

        job = self.job_manager.cancel(job_id)

        from medds_agent.job_manager import STATUS_CANCELLED
        if job.status == STATUS_CANCELLED:
            return (
                f"Job '{job_id}' has been cancelled. "
                f"The executor has been restarted. "
                f"Note: any variables defined in the cancelled execution are lost."
            )
        else:
            return (
                f"Job '{job_id}' could not be cancelled (status: {job.status}). "
                f"It may have already finished."
            )

    def get_title(self, args: Dict) -> str:
        job_id = args.get("job_id", "")
        return f"Cancel job {job_id}" if job_id else "Cancel job"

    def get_tool_call_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The job_id to cancel.",
                        },
                    },
                    "required": ["job_id"],
                },
            },
        }


class FinalResponseTool(Tool):
    """
    Internal tool that signals the agent has finished its analysis and is ready
    to deliver a final answer to the user.

    This tool is automatically injected into every agent by Agent.__init__.
    It must be called alone (not alongside other tools). Calling it ends the
    current round and delivers `response` as the agent's response.
    """

    def __init__(self):
        super().__init__(
            name="final_response",
            description=(
                "Call this tool when you have are ready to end the current round and deliver "
                "your complete response to the user. This must be the only tool call in your "
                "response — do not mix it with other tool calls."
            )
        )

    def execute(self, param: Dict) -> str:
        # Never executed directly; the agent intercepts this tool call.
        return param.get("response", "")

    def get_tool_call_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "Your complete final response to the user.",
                        },
                    },
                    "required": ["response"],
                },
            },
        }

    def get_title(self, args: Dict) -> str:
        return "Final Response"