"""
FastAPI server for the Medical Data Science Agent (MedDSAgent).

This module provides a pure REST API backend. It does not serve any
frontend assets — those live in separate projects (MedDSAgent-App,
MedDSAgent-VSCode, etc.) and consume these endpoints.
"""
import os
import shutil
import asyncio
import aiofiles
import zipfile
import io
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Body
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from medds_agent.manager import SessionManager
from medds_agent.tools import PythonExecutorTool, RExecutorTool
from medds_agent.document_parser import DocumentParser, is_parseable

# =============================================================================
# Configuration & Global State
# =============================================================================

class ServerState:
    def __init__(self):
        self.work_dir = os.environ.get("WORK_DIR", "./workspace")
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", 7842))
        self.manager: Optional[SessionManager] = None
        self.document_parser: Optional[DocumentParser] = None

state = ServerState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the SessionManager on startup."""
    print(f"Starting MedDSAgent Server...")
    print(f"Workspace Directory: {os.path.abspath(state.work_dir)}")
    
    # Ensure root workspace exists
    os.makedirs(state.work_dir, exist_ok=True)
    
    # Initialize the Persistent Session Manager
    state.manager = SessionManager(work_dir=state.work_dir)

    # Initialize the Document Parser (shares the same DB)
    state.document_parser = DocumentParser(db=state.manager.db)

    yield

    print("Shutting down...")

app = FastAPI(
    title="MedDSAgent API",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Health Check
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """API root."""
    return {"status": "ok", "service": "MedDSAgent"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for docker-compose, VS Code extension, and monitoring."""
    return {"status": "ok", "service": "MedDSAgent", "port": state.port}

# =============================================================================
# Specialty Prompt Endpoints
# =============================================================================

SPECIALTY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset", "prompt_templates", "specialty")

@app.get("/specialty-prompts", tags=["Specialty"])
async def list_specialty_prompts():
    """Return the index of available pre-defined specialty prompts."""
    index_path = os.path.join(SPECIALTY_DIR, "index.json")
    if not os.path.exists(index_path):
        return []
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/specialty-prompts/{prompt_id}", tags=["Specialty"])
async def get_specialty_prompt(prompt_id: str):
    """Return the content of a specific specialty prompt by its ID."""
    index_path = os.path.join(SPECIALTY_DIR, "index.json")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Specialty index not found")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    entry = next((e for e in index if e["id"] == prompt_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Specialty prompt not found")
    file_path = os.path.join(SPECIALTY_DIR, entry["filename"])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Specialty prompt file not found")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"id": entry["id"], "display_name": entry["display_name"], "content": content}

# =============================================================================
# Pydantic Models
# =============================================================================

class SessionConfig(BaseModel):
    # Common / Shared
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None

    # Sampling Parameters
    temperature: float = 1.0
    top_p: float = 1.0

    # Azure Specific
    llm_api_version: Optional[str] = None

    # Database Connection Code (Python code that creates db_engine or conn)
    db_connection_code: Optional[str] = None

    # Language: "python" or "r"
    language: str = "python"

    # Reasoning Effort: "low", "medium", "high", or None (not applicable)
    reasoning_effort: Optional[str] = None

    # Specialty: selected pre-defined specialty ID (for dropdown restore)
    specialty_id: Optional[str] = None

    # Specialty Prompt: domain/task-specific instructions appended to system prompt
    specialty_prompt: Optional[str] = None

class CreateSessionRequest(BaseModel):
    name: str
    config: SessionConfig

class UpdateSessionRequest(BaseModel):
    name: str
    config: SessionConfig

class ChatRequest(BaseModel):
    message: str
    stream: bool = False

class TestConnectionRequest(BaseModel):
    code: str

class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    size_human: str
    is_directory: bool
    modified_at: datetime

# =============================================================================
# Session Management Endpoints
# =============================================================================

@app.get("/sessions", tags=["Sessions"])
async def list_sessions():
    return await state.manager.list_sessions()

@app.post("/sessions", tags=["Sessions"])
async def create_session(request: CreateSessionRequest):
    try:
        config_dict = request.config.model_dump(exclude_none=True)
        session_id = await state.manager.create_session(
            name=request.name,
            config=config_dict
        )
        return {"session_id": session_id, "name": request.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Get full session details including configuration."""
    try:
        config = await state.manager.get_session_config(session_id)
        if config is None:
            raise HTTPException(status_code=404, detail="Session not found")
            
        sessions = await state.manager.list_sessions()
        session_info = next((s for s in sessions if s['session_id'] == session_id), None)
        
        if not session_info:
             raise HTTPException(status_code=404, detail="Session not found")
             
        return {
            "session_id": session_id,
            "name": session_info['name'],
            "last_accessed": session_info['last_accessed'],
            "config": config
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/sessions/{session_id}", tags=["Sessions"])
async def update_session(session_id: str, request: UpdateSessionRequest):
    """Update session name and configuration."""
    try:
        config_dict = request.config.model_dump(exclude_none=True)
        await state.manager.update_session(session_id, request.name, config_dict)
        return {"status": "updated", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    try:
        await state.manager.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/sessions/{session_id}/name", tags=["Sessions"])
async def rename_session(session_id: str, name: str = Body(..., embed=True)):
    try:
        await state.manager.rename_session(session_id, name)
        return {"status": "renamed", "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-db-connection", tags=["Sessions"])
async def test_db_connection(request: TestConnectionRequest):
    """
    Test database connection code.
    Executes the provided Python code in an isolated namespace and checks
    if it creates a 'db_engine' or 'conn' object.
    """
    import sys
    import io

    code = request.code
    if not code or not code.strip():
        raise HTTPException(status_code=400, detail="No connection code provided")

    # Create isolated namespace with common imports
    test_globals = {"__builtins__": __builtins__}
    test_locals = {}

    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = io.StringIO()
    sys.stderr = captured_stderr = io.StringIO()

    try:
        # Execute the connection code
        exec(code, test_globals, test_locals)
        test_globals.update(test_locals)

        # Check for connection objects
        db_engine = test_globals.get('db_engine')
        conn = test_globals.get('conn')

        if db_engine is None and conn is None:
            raise HTTPException(
                status_code=400,
                detail="Connection code must create either 'db_engine' or 'conn' variable"
            )

        # Try to validate the connection
        connection_type = None
        if db_engine is not None:
            connection_type = "db_engine (SQLAlchemy)"
            # Try a simple connection test
            try:
                with db_engine.connect() as test_conn:
                    pass
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"SQLAlchemy connection failed: {str(e)}")

        if conn is not None:
            connection_type = "conn (Direct connection)"
            # Try a simple test - just check if it's a valid connection object
            if not hasattr(conn, 'cursor') and not hasattr(conn, 'execute'):
                raise HTTPException(
                    status_code=400,
                    detail="'conn' object doesn't appear to be a valid database connection"
                )

        return {
            "status": "success",
            "message": f"Connection successful! Found: {connection_type}",
            "connection_type": connection_type
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=400, detail=f"Connection test failed: {error_detail}")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Clean up connections
        try:
            if 'db_engine' in test_globals and test_globals['db_engine'] is not None:
                test_globals['db_engine'].dispose()
        except:
            pass
        try:
            if 'conn' in test_globals and test_globals['conn'] is not None:
                test_globals['conn'].close()
        except:
            pass

# =============================================================================
# Chat & History Endpoints
# =============================================================================

@app.get("/sessions/{session_id}/history", tags=["Chat"])
async def get_history(session_id: str):
    try:
        steps = state.manager.db.get_history_steps(session_id)
        serialized_steps = []
        for step in steps:
            data = step.serialize()
            data['type'] = step.__class__.__name__
            serialized_steps.append(data)
        return {"steps": serialized_steps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.post("/sessions/{session_id}/chat", tags=["Chat"])
async def chat(session_id: str, request: ChatRequest):
    if request.stream:
        return await _chat_stream(session_id, request.message)
    else:
        try:
            await state.manager.run_session(session_id, request.message, stream=False)
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/stop", tags=["Chat"])
async def stop_chat(session_id: str):
    """Stop the current generation."""
    try:
        success = await state.manager.stop_session(session_id)
        if success:
            return {"status": "stopped"}
        else:
            return {"status": "no_active_run"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _chat_stream(session_id: str, message: str):
    async def _get_env_snapshot() -> Optional[dict]:
        """Collect current variable state for pushing through the SSE stream."""
        try:
            agent = await state.manager.get_agent(session_id)
            if not agent:
                return None
            variables: Dict[str, Any] = {}
            python_tool = next((t for t in agent.tools if isinstance(t, PythonExecutorTool)), None)
            r_tool = next((t for t in agent.tools if isinstance(t, RExecutorTool)), None)
            if python_tool and hasattr(python_tool, 'get_state'):
                variables["python"] = python_tool.get_state()
                variables["language"] = "python"
            elif r_tool and hasattr(r_tool, 'get_state'):
                variables["r"] = r_tool.get_state()
                variables["language"] = "r"
            return variables
        except Exception:
            return None

    async def event_generator():
        import json
        try:
            stream = await state.manager.run_session(session_id, message, stream=True)
            async for chunk in stream:
                yield f"data: {json.dumps(chunk)}\n\n"

                # After each tool execution, push environment snapshot
                if chunk.get('type') == 'tool_output':
                    env = await _get_env_snapshot()
                    if env is not None:
                        yield f"data: {json.dumps({'type': 'env_update', 'data': env})}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            # -------------------------------------------------------

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# =============================================================================
# Environment & State Inspection
# =============================================================================

@app.get("/sessions/{session_id}/variables", tags=["Environment"])
async def get_variables(session_id: str):
    try:
        agent = await state.manager.get_agent(session_id)
        if not agent:
             raise HTTPException(status_code=404, detail="Session not found")

        variables = {}
        python_tool = next((t for t in agent.tools if isinstance(t, PythonExecutorTool)), None)
        r_tool = next((t for t in agent.tools if isinstance(t, RExecutorTool)), None)

        if python_tool and hasattr(python_tool, 'get_state'):
            variables["python"] = python_tool.get_state()
            variables["language"] = "python"
        elif r_tool and hasattr(r_tool, 'get_state'):
            variables["r"] = r_tool.get_state()
            variables["language"] = "r"

        return variables
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Memory Inspection
# =============================================================================

@app.get("/sessions/{session_id}/memory", tags=["Memory"])
async def get_memory(session_id: str):
    """Inspect the exact messages the LLM receives from the agent's memory."""
    try:
        agent = await state.manager.get_agent(session_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Session not found")
        return agent.agent_memory.get_memory_debug(agent.history)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/memory/compression", tags=["Memory"])
async def get_memory_compression(session_id: str):
    """Inspect compressed vs original content for each step in memory."""
    try:
        agent = await state.manager.get_agent(session_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Session not found")
        return agent.agent_memory.get_compression_debug(agent.history)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# File Management
# =============================================================================

def _get_session_path(session_id: str, subpath: str = "") -> str:
    session_dir = os.path.join(state.manager.sessions_dir, session_id)
    if not os.path.exists(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")
    
    target_path = os.path.abspath(os.path.join(session_dir, subpath))
    if not target_path.startswith(session_dir):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return target_path

def _format_size(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != 'B' else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

@app.get("/sessions/{session_id}/files", response_model=List[FileInfo], tags=["Files"])
async def list_files(session_id: str, path: str = Query(default="")):
    try:
        target_path = _get_session_path(session_id, path)
        if not os.path.exists(target_path):
             return []
        
        files = []
        for entry in sorted(os.listdir(target_path)):
            if entry.startswith('.') or entry in ["state.pkl", "state.RData", "session_config.json"]:
                continue
                
            full_path = os.path.join(target_path, entry)
            stat = os.stat(full_path)
            # Return path relative to session root, not the queried path
            rel_path = os.path.relpath(full_path, os.path.join(state.manager.sessions_dir, session_id))
            
            files.append(FileInfo(
                name=entry,
                path=rel_path,
                size=stat.st_size,
                size_human=_format_size(stat.st_size),
                is_directory=os.path.isdir(full_path),
                modified_at=datetime.fromtimestamp(stat.st_mtime)
            ))
        return files
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/files", tags=["Files"])
async def upload_file(session_id: str, file: UploadFile = File(...), path: str = Query(default="uploads")):
    try:
        target_dir = _get_session_path(session_id, path)
        os.makedirs(target_dir, exist_ok=True)
        final_path = os.path.join(target_dir, file.filename)

        # Stage file: write to a temp location first so the file is not
        # visible in uploads/ until parsing is complete.
        session_dir = os.path.join(state.manager.sessions_dir, session_id)
        staging_dir = os.path.join(session_dir, ".staging")
        os.makedirs(staging_dir, exist_ok=True)
        staging_path = os.path.join(staging_dir, file.filename)

        async with aiofiles.open(staging_path, 'wb') as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)

        # Parse document if it's a supported type (runs synchronously — user
        # perceives this as part of the upload).  Non-parseable files skip
        # this step and go straight to uploads/.
        parsed = False
        if is_parseable(staging_path):
            parsed = await asyncio.get_event_loop().run_in_executor(
                None,
                state.document_parser.parse,
                session_id,
                staging_path,
                file.filename,
            )

        # Move from staging to final destination
        shutil.move(staging_path, final_path)

        # Build system message
        if parsed:
            system_msg = (
                f"File '{file.filename}' uploaded to '{path}'. "
                f"This document has been parsed and indexed. "
                f"Use DocumentSearch tool to browse its sections."
            )
        else:
            system_msg = f"File '{file.filename}' uploaded to '{path}'."

        await state.manager.add_system_message(session_id, system_msg)
        return {"status": "uploaded", "filename": file.filename, "parsed": parsed}
    except Exception as e:
        # Clean up staging file on error
        if 'staging_path' in locals() and os.path.exists(staging_path):
            os.remove(staging_path)
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# File Content Read/Write (Editor Support)
# IMPORTANT: These routes must be defined BEFORE the catch-all {file_path:path} routes
# =============================================================================

# File extensions that can be edited as text
TEXT_EXTENSIONS = {
    '.py', '.r', '.R', '.sql', '.txt', '.md', '.json', '.csv', '.tsv',
    '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.sh', '.bash',
    '.js', '.ts', '.html', '.css', '.xml', '.log', '.env', '.gitignore',
    '.data'  # For .data files like breast-cancer.data.csv
}

# File extensions for images (read-only preview)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}

# Max lines to return for large files
MAX_PREVIEW_LINES = 1000

class FileContentResponse(BaseModel):
    content: str
    file_type: str  # 'text', 'image', 'binary'
    is_truncated: bool
    total_lines: Optional[int] = None
    extension: str

class SaveFileRequest(BaseModel):
    content: str

@app.get("/sessions/{session_id}/files/{file_path:path}/content", tags=["Files"])
async def get_file_content(session_id: str, file_path: str):
    """Get file content for editing. Returns first 1000 lines for text files."""
    target_path = _get_session_path(session_id, file_path)

    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="File not found")

    if os.path.isdir(target_path):
        raise HTTPException(status_code=400, detail="Cannot read directory content")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Handle images - return base64 for preview
    if ext in IMAGE_EXTENSIONS:
        import base64
        try:
            async with aiofiles.open(target_path, 'rb') as f:
                data = await f.read()
            b64_content = base64.b64encode(data).decode('utf-8')
            mime_type = f"image/{ext[1:]}"
            if ext == '.svg':
                mime_type = "image/svg+xml"
            return FileContentResponse(
                content=f"data:{mime_type};base64,{b64_content}",
                file_type="image",
                is_truncated=False,
                extension=ext
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read image: {str(e)}")

    # Handle text files
    if ext in TEXT_EXTENSIONS or ext == '':
        try:
            async with aiofiles.open(target_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = await f.readlines()

            total_lines = len(lines)
            is_truncated = total_lines > MAX_PREVIEW_LINES

            if is_truncated:
                content = ''.join(lines[:MAX_PREVIEW_LINES])
            else:
                content = ''.join(lines)

            return FileContentResponse(
                content=content,
                file_type="text",
                is_truncated=is_truncated,
                total_lines=total_lines,
                extension=ext
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

    # Binary files - not editable
    return FileContentResponse(
        content="",
        file_type="binary",
        is_truncated=False,
        extension=ext
    )

@app.put("/sessions/{session_id}/files/{file_path:path}/content", tags=["Files"])
async def save_file_content(session_id: str, file_path: str, request: SaveFileRequest):
    """Save file content and notify agent via system message."""
    target_path = _get_session_path(session_id, file_path)

    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="File not found")

    if os.path.isdir(target_path):
        raise HTTPException(status_code=400, detail="Cannot write to directory")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Only allow saving text files
    if ext not in TEXT_EXTENSIONS and ext != '':
        raise HTTPException(status_code=400, detail="Cannot edit this file type")

    try:
        async with aiofiles.open(target_path, 'w', encoding='utf-8') as f:
            await f.write(request.content)

        # Add system message to notify agent
        await state.manager.add_system_message(
            session_id,
            f"User edited file '{file_path}'."
        )

        return {"status": "saved", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

# =============================================================================
# File Download/Delete (catch-all routes - must be AFTER /content routes)
# =============================================================================

@app.get("/sessions/{session_id}/files/{file_path:path}", tags=["Files"])
async def download_file(session_id: str, file_path: str):
    target_path = _get_session_path(session_id, file_path)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Handle Directory Download (Zip)
    if os.path.isdir(target_path):
        try:
            # Create a zip file in memory
            mem_zip = io.BytesIO()
            with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                # Walk the directory
                parent_dir = os.path.dirname(target_path)
                for root, dirs, files in os.walk(target_path):
                    for file in files:
                        abs_file_path = os.path.join(root, file)
                        # Archive name should be relative to the parent of the target folder
                        # so the zip contains the target folder name as the root
                        arcname = os.path.relpath(abs_file_path, parent_dir)
                        zf.write(abs_file_path, arcname=arcname)

            mem_zip.seek(0)
            filename = f"{os.path.basename(target_path)}.zip"
            return StreamingResponse(
                iter([mem_zip.getvalue()]),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to zip directory: {str(e)}")

    # Handle Single File Download
    return FileResponse(target_path, filename=os.path.basename(target_path), media_type="application/octet-stream")

@app.delete("/sessions/{session_id}/files/{file_path:path}", tags=["Files"])
async def delete_file(session_id: str, file_path: str):
    target_path = _get_session_path(session_id, file_path)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        is_directory = os.path.isdir(target_path)
        if is_directory:
            shutil.rmtree(target_path)
        else:
            os.remove(target_path)

        # Add system step to history
        item_type = "folder" if is_directory else "file"
        await state.manager.add_system_message(
            session_id,
            f"User deleted {item_type} '{file_path}'."
        )

        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn
    reload = os.environ.get("RELOAD", "false").lower() in ("true", "1", "yes")
    uvicorn.run("medds_agent.server:app", host=state.host, port=state.port, reload=reload)

if __name__ == "__main__":
    main()