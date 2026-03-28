import os
import sys
import uuid
import asyncio
import shutil
import json
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

from medds_agent.agents import Agent
from medds_agent.history import History, SystemStep
from medds_agent.memory import SlideWindowAgentMemory, IndexedAgentMemory
from medds_agent.database import InternalDatabase
from medds_agent.tools import PythonExecutorTool, RExecutorTool, FileSystemTool, DocumentSearchTool
from medds_agent.workers import CodeWorker, WorkerStartupError
from medds_agent.job_manager import JobManager

# Import Engines for dynamic instantiation
from medds_agent.engines import (
    OpenAIInferenceEngine,
    AzureOpenAIInferenceEngine,
    VLLMInferenceEngine,
    SGLangInferenceEngine,
    OpenRouterInferenceEngine,
    AgentBasicLLMConfig,
    AgentReasoningLLMConfig
)

class SessionManager:
    """
    Manages the lifecycle of Agent sessions with dynamic configuration.

    Architecture:
    - Hot Cache: Keeps active Agent instances in memory.
    - Persistent Config: Saves LLM/DB settings to the internal database.
    - Isolation: Each session has its own directory and isolated state.
    """
    
    def __init__(self, work_dir: str):
        """
        Parameters:
        -----------
        work_dir : str
            Root directory for all session data.
        """
        self.root_work_dir = os.path.abspath(work_dir)
        self.sessions_dir = os.path.join(self.root_work_dir, "sessions")
        self.db_path = os.path.join(self.root_work_dir, "internal.db")
        
        # Ensure directories exist
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # Initialize Database
        self.db = InternalDatabase(self.db_path)
        
        # In-memory cache: {session_id: {"agent": Agent, "worker": CodeWorker,
        #                                "last_active": datetime, "persisted_steps": int}}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
        # Active Task Tracking for Cancellation
        self._active_tasks: Dict[str, asyncio.Task] = {}

    async def create_session(self, name: str, config: Dict[str, Any]) -> str:
        """
        Create a new session with specific LLM and Tool configurations.
        """
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(self.sessions_dir, session_id)
        
        # 1. Setup File System
        os.makedirs(os.path.join(session_dir, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "outputs"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "scripts"), exist_ok=True)
        os.makedirs(os.path.join(session_dir, "internal"), exist_ok=True)
        
        # 2. Create DB Entry with config
        self.db.create_session(session_id, name, session_dir, config=config)

        # 2b. Save specialty to dedicated columns
        self.db.save_session_specialty(
            session_id,
            specialty_id=config.get("specialty_id"),
            specialty_prompt=config.get("specialty_prompt")
        )

        # 3. Initialize Agent
        try:
            agent, connection_info, worker, indexer_worker, indexer_jm = self._init_agent_instance(session_dir, config, session_id=session_id)
        except Exception as e:
            # Rollback if init fails
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
            self.db.delete_session(session_id)
            raise e

        # 4. Add system step for DB connection (if established)
        if connection_info:
            system_step = SystemStep(
                start_time=datetime.now(),
                end_time=datetime.now(),
                system_message=f"Database connection established. Use variable '{connection_info['variable']}' ({connection_info['type']}) to query the database."
            )
            agent.history.add_step(system_step)
            agent.agent_memory.on_step_added(system_step)

        # 5. Cache it (worker is stored in the cache entry for lifecycle management)
        async with self._lock:
            self._cache[session_id] = {
                "agent": agent,
                "worker": worker,
                "indexer_worker": indexer_worker,
                "indexer_job_manager": indexer_jm,
                "last_active": datetime.now(),
                "persisted_steps": 0,
            }

        # 6. Save session to persist the system step to database
        if connection_info:
            await self.save_session(session_id)

        return session_id

    async def get_agent(self, session_id: str) -> Optional[Agent]:
        """Retrieve an agent (from Cache or Disk)."""
        async with self._lock:
            if session_id in self._cache:
                self._cache[session_id]["last_active"] = datetime.now()
                self.db.update_last_accessed(session_id)
                return self._cache[session_id]["agent"]
        
        return await self._load_session_from_storage(session_id)
    
    async def add_system_message(self, session_id: str, system_message: str):
        """Add a system message step to the session's history."""
        agent = await self.get_agent(session_id)
        if not agent:
            raise ValueError("Session not found")
        
        system_step = SystemStep(
            start_time=datetime.now(),
            end_time=datetime.now(),
            system_message=system_message
        )
        agent.history.add_step(system_step)
        agent.agent_memory.on_step_added(system_step)
        await self.save_session(session_id)

    async def get_session_config(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the configuration for a specific session."""
        config = self.db.get_session_config(session_id)
        if config is not None:
            return config

        return {}

    async def update_session(self, session_id: str, name: str, config: Dict[str, Any]):
        """Update session name and configuration in-place, preserving worker state."""
        # 1. Get old config before any writes
        old_config = self.db.get_session_config(session_id) or {}

        # 2. Guard: language/python_bin/r_home are immutable after session creation
        immutable_keys = ("language", "python_bin", "r_home")
        for key in immutable_keys:
            old_val = (old_config.get(key) or "").strip()
            new_val = (config.get(key) or "").strip()
            if old_val and new_val and old_val != new_val:
                raise ValueError(
                    f"Cannot change '{key}' after session creation. "
                    f"Current value: '{old_val}', requested value: '{new_val}'."
                )

        # 3. Update Name and Config in DB
        self.db.rename_session(session_id, name)
        self.db.save_session_config(session_id, config)
        self.db.save_session_specialty(
            session_id,
            specialty_id=config.get("specialty_id"),
            specialty_prompt=config.get("specialty_prompt")
        )

        # 4. If session is not cached, nothing more to do — next load will use new DB config
        async with self._lock:
            cached = session_id in self._cache
            if cached:
                cache_entry = self._cache[session_id]
                agent = cache_entry["agent"]
                worker = cache_entry.get("worker")

        if not cached:
            return

        language = config.get("language", "python").lower()

        # 5. LLM config update — swap engine in-place
        llm_keys = {"llm_provider", "llm_model", "llm_api_key", "llm_base_url",
                    "llm_api_version", "temperature", "top_p", "reasoning_effort"}
        if any(old_config.get(k) != config.get(k) for k in llm_keys):
            new_engine = self._build_llm_engine(config)
            agent.llm_engine = new_engine
            agent.agent_memory._llm_engine = new_engine

        # 6. Specialty prompt update
        if (old_config.get("specialty_prompt") != config.get("specialty_prompt") or
                old_config.get("specialty_id") != config.get("specialty_id")):
            agent.agent_memory.set_system_prompt(agent.system_prompt, config.get("specialty_prompt"))

        # 7. DB connection code update — inject into existing worker (no restart)
        old_db_code = (old_config.get("db_connection_code") or "").strip()
        new_db_code = (config.get("db_connection_code") or "").strip()
        if new_db_code and new_db_code != old_db_code:
            if worker:
                try:
                    result = worker.send_command_sync("inject", {"code": new_db_code})
                    if result.get("error"):
                        print(f"Warning: DB connection code injection error: {result['error']}")
                    else:
                        connection_info = self._get_connection_info_from_worker(worker, language)
                        if connection_info:
                            await self.add_system_message(
                                session_id,
                                f"Database connection established. Use variable '{connection_info['variable']}' ({connection_info['type']}) to query the database."
                            )
                except Exception as e:
                    print(f"Error injecting DB connection code: {e}")

    async def run_session(self, session_id: str, user_input: str, stream: bool = False):
        """Run the agent and immediately persist state/history, with cancellation support.

        Uses async agent methods to avoid blocking the event loop.
        """
        agent = await self.get_agent(session_id)
        if not agent:
             raise ValueError("Session not found")

        # Define saving callback
        async def save_callback():
            await self.save_session(session_id)

        # Create the task wrapper logic
        async def task_wrapper():
            try:
                if stream:
                    # Use async streaming - no event loop blocking
                    async for chunk in agent.run_event_stream_async(user_input):
                        yield chunk
                else:
                    # Use async non-streaming
                    await agent.run_async(user_input)
            except asyncio.CancelledError:
                # Handle forced stop
                print(f"Session {session_id} task cancelled.")
                await self.add_system_message(
                    session_id,
                    "The session was cancelled by the user."
                )
                raise # Re-raise to ensure proper cleanup if needed
            finally:
                await save_callback()
                # Remove from active tasks when done
                self._active_tasks.pop(session_id, None)

        # Register the task
        current_task = asyncio.current_task()
        if current_task:
            self._active_tasks[session_id] = current_task

        return task_wrapper()

    async def stop_session(self, session_id: str):
        """Cancels the running task for the session."""
        task = self._active_tasks.get(session_id)
        if task:
            task.cancel()
            return True
        return False

    async def save_session(self, session_id: str):
        """Persist Python state and append new history steps."""
        async with self._lock:
            if session_id not in self._cache:
                return
            
            cache_entry = self._cache[session_id]
            agent = cache_entry["agent"]
            persisted_steps_count = cache_entry["persisted_steps"]
            
            # 1. Serialize agent memory state
            agent_memory_data = agent.agent_memory.serialize()

            # 2. Save Executor State via worker subprocess
            session_dir = os.path.join(self.sessions_dir, session_id)
            worker = cache_entry.get("worker")
            python_tool = next((t for t in agent.tools if isinstance(t, PythonExecutorTool)), None)
            r_tool = next((t for t in agent.tools if isinstance(t, RExecutorTool)), None)

            if python_tool and worker and worker.is_alive:
                state_path = os.path.join(session_dir, "state.pkl")
                try:
                    worker.send_command_sync("save_state", {"path": state_path})
                    self.db.save_session_state(
                        session_id=session_id,
                        agent_memory_data=agent_memory_data,
                        python_state_path=f"sessions/{session_id}/state.pkl"
                    )
                except Exception as e:
                    print(f"Error saving python state for {session_id}: {e}")

            elif r_tool and worker and worker.is_alive:
                r_state_path = os.path.join(session_dir, "state.RData")
                try:
                    worker.send_command_sync("save_state", {"path": r_state_path})
                    self.db.save_session_state(
                        session_id=session_id,
                        agent_memory_data=agent_memory_data,
                        python_state_path=f"sessions/{session_id}/state.RData"
                    )
                except Exception as e:
                    print(f"Error saving R state for {session_id}: {e}")

            # 2. Save NEW History Steps
            all_steps = []
            for r in agent.history.rounds:
                all_steps.extend(r.steps)
            
            total_steps = len(all_steps)
            if total_steps > persisted_steps_count:
                # Add only new steps
                current_idx = 0
                for r in agent.history.rounds:
                    for step in r.steps:
                        if current_idx >= persisted_steps_count:
                            self.db.add_history_step(session_id, r.round_num, step)
                        current_idx += 1
                
                cache_entry["persisted_steps"] = total_steps

            # 3. Update Timestamp
            self.db.update_last_accessed(session_id)

    def _get_connection_info_from_worker(self, worker: "CodeWorker", language: str) -> Optional[Dict[str, str]]:
        """
        Query the worker's variable state to detect DB connection objects.
        Works for both Python (db_engine / conn) and R (DBI connection classes).
        """
        try:
            data = worker.send_command_sync("get_state", {})
            variables = data.get("variables", [])
        except Exception as e:
            print(f"Warning: Could not query worker state for connection info: {e}")
            return None

        if language == "r":
            dbi_classes = {
                "PqConnection", "MariaDBConnection", "MySQLConnection",
                "OraConnection", "SQLiteConnection", "OdbcConnection", "DBIConnection"
            }
            for var in variables:
                var_type = var.get("type", "")
                if any(cls in var_type for cls in dbi_classes) or "Connection" in var_type:
                    return {"variable": var["name"], "type": var_type}
        else:
            # Python: look for db_engine or conn by name
            connection_info = None
            for var in variables:
                name = var.get("name", "")
                var_type = var.get("type", "")
                if name == "db_engine":
                    connection_info = {"variable": "db_engine", "type": f"SQLAlchemy ({var_type})"}
                elif name == "conn" and connection_info is None:
                    connection_info = {"variable": "conn", "type": var_type}
            return connection_info

        return None

    async def _load_session_from_storage(self, session_id: str) -> Optional[Agent]:
        """Reconstruct agent from DB config, DB history, and state.pkl."""
        
        # 1. Get Metadata & Config
        session_info = self.db.get_session(session_id)
        if not session_info:
            return None
        
        session_dir = session_info['work_dir']
        config = self.db.get_session_config(session_id)
        
        # 2. Initialize Agent using saved Config (connection_info ignored for reload - already in history)
        try:
            agent, _, worker, indexer_worker, indexer_jm = self._init_agent_instance(session_dir, config, session_id=session_id)
        except Exception as e:
            print(f"Error initializing agent for {session_id}: {e}")
            return None

        # 3. Reconstruct History
        history_steps = self.db.get_history_steps(session_id)
        history = History()
        for step in history_steps:
            history.add_step(step)
        agent.history = history

        # 4. Load Executor State into the worker subprocess
        python_tool = next((t for t in agent.tools if isinstance(t, PythonExecutorTool)), None)
        r_tool = next((t for t in agent.tools if isinstance(t, RExecutorTool)), None)

        if python_tool and worker and worker.is_alive:
            state_path = os.path.join(session_dir, "state.pkl")
            if os.path.exists(state_path):
                try:
                    worker.send_command_sync("load_state", {"path": state_path})
                except Exception as e:
                    print(f"Warning: Failed to load Python state for {session_id}: {e}")

        elif r_tool and worker and worker.is_alive:
            r_state_path = os.path.join(session_dir, "state.RData")
            if os.path.exists(r_state_path):
                try:
                    worker.send_command_sync("load_state", {"path": r_state_path})
                except Exception as e:
                    print(f"Warning: Failed to load R state for {session_id}: {e}")

        # 5. Restore Agent Memory State
        session_state = self.db.get_session_state(session_id)
        if session_state and session_state.get('agent_memory_data'):
            try:
                agent.agent_memory.deserialize(session_state['agent_memory_data'])
            except Exception as e:
                print(f"Warning: Failed to restore agent memory for {session_id}: {e}")

        # 6. Cache (include workers for lifecycle management)
        total_steps = sum(len(r.steps) for r in history.rounds)
        async with self._lock:
            self._cache[session_id] = {
                "agent": agent,
                "worker": worker,
                "indexer_worker": indexer_worker,
                "indexer_job_manager": indexer_jm,
                "last_active": datetime.now(),
                "persisted_steps": total_steps,
            }

        return agent

    def _init_agent_instance(self, session_dir: str, config: Dict[str, Any], session_id: str = None) -> Tuple[Agent, Optional[Dict[str, str]], "CodeWorker", "CodeWorker", "JobManager"]:
        """
        Construct Agent based on dynamic configuration.

        Returns:
            Tuple of (Agent, connection_info, worker) where connection_info is None if no DB connection.
        """

        # 1. Spawn the appropriate subprocess worker
        language = config.get("language", "python").lower()
        connection_info = None

        if language == "r":
            handler_class_path = "medds_agent.worker_handlers.RHandler"
        else:
            handler_class_path = "medds_agent.worker_handlers.PythonHandler"

        python_bin = config.get("python_bin") or os.environ.get("MEDDS_PYTHON_BIN") or sys.executable

        worker_env = {}
        r_home = config.get("r_home") or os.environ.get("MEDDS_R_HOME")
        if r_home:
            worker_env["R_HOME"] = r_home

        worker = CodeWorker(
            handler_class_path=handler_class_path,
            python_bin=python_bin,
            handler_kwargs={"work_dir": session_dir},
            env=worker_env or None,
        )
        job_manager = JobManager(worker=worker)
        ready_info = worker.ready_info

        # 2. Configure Tools
        if language == "r":
            executor_tool = RExecutorTool(job_manager=job_manager, ready_info=ready_info)
        else:
            executor_tool = PythonExecutorTool(job_manager=job_manager, ready_info=ready_info)

        tools = [
            executor_tool,
            FileSystemTool(dir=session_dir),
        ]

        # Add DocumentSearchTool if session_id is available
        if session_id:
            uploads_dir = os.path.join(session_dir, "uploads")
            tools.append(DocumentSearchTool(session_id=session_id, db=self.db, uploads_dir=uploads_dir))

        # Inject DB connection code into the worker (replaces in-process exec)
        db_connection_code = config.get("db_connection_code", "")
        if db_connection_code and db_connection_code.strip():
            try:
                result = worker.send_command_sync("inject", {"code": db_connection_code})
                if result.get("error"):
                    print(f"Warning: DB connection code injection error: {result['error']}")
                else:
                    # Determine connection variable info by querying worker state
                    connection_info = self._get_connection_info_from_worker(worker, language)
            except Exception as e:
                print(f"Error injecting DB connection code: {e}")
            
        # 2. Configure LLM Engine
        llm_engine = self._build_llm_engine(config)

        # 3. Create Agent Memory
        start_window_size = int(os.environ.get("MEMORY_START_WINDOW_SIZE", "5"))
        recent_window_size = int(os.environ.get("MEMORY_RECENT_WINDOW_SIZE", "20"))
        compress_threshold = int(os.environ.get("MEMORY_COMPRESS_THRESHOLD", "2000"))
        agent_memory = IndexedAgentMemory(
            llm_engine=llm_engine,
            compress_threshold=compress_threshold,
            recent_window_size=recent_window_size,
            start_window_size=start_window_size,
        )

        # 4. Spawn the document indexer worker (subprocess; lazy-loads docling on first use)
        indexer_worker = CodeWorker(
            handler_class_path="medds_agent.worker_handlers.DocumentIndexerHandler",
            python_bin=python_bin,
            handler_kwargs={"db_path": self.db_path},
            env=worker_env or None,
        )
        indexer_jm = JobManager(worker=indexer_worker)

        # 5. Create Agent
        agent = Agent(
            llm_engine=llm_engine,
            history=History(),
            agent_memory=agent_memory,
            specialty_prompt=config.get("specialty_prompt", None),
            tools=tools,
            verbose=False
        )
        return agent, connection_info, worker, indexer_worker, indexer_jm

    def _build_llm_engine(self, config: Dict[str, Any]):
        """Construct and return an LLM inference engine from a session config dict."""
        provider = config.get("llm_provider", "openai").lower()
        model = config.get("llm_model", "gpt-4")
        api_key = config.get("llm_api_key")
        base_url = config.get("llm_base_url")

        temperature = float(config.get("temperature", 1.0))
        top_p = float(config.get("top_p", 1.0))
        max_new_tokens = int(os.environ.get("LLM_MAX_NEW_TOKENS", "16384"))
        max_input_tokens = int(os.environ.get("LLM_MAX_INPUT_TOKENS", "120000"))

        reasoning_effort = config.get("reasoning_effort")
        if reasoning_effort and reasoning_effort in ["low", "medium", "high"]:
            llm_config = AgentReasoningLLMConfig(max_input_tokens=max_input_tokens)
        else:
            llm_config = AgentBasicLLMConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                max_input_tokens=max_input_tokens
            )

        engine_kwargs = {"model": model, "config": llm_config}

        if api_key:
            engine_kwargs["api_key"] = api_key
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            if provider == "azure":
                os.environ["AZURE_OPENAI_API_KEY"] = api_key

        if base_url:
            if provider == "azure":
                engine_kwargs["azure_endpoint"] = base_url
            else:
                engine_kwargs["base_url"] = base_url

        if provider == "openai":
            return OpenAIInferenceEngine(**engine_kwargs)
        elif provider == "azure":
            engine_kwargs["api_version"] = config.get("llm_api_version") or "2023-12-01-preview"
            return AzureOpenAIInferenceEngine(**engine_kwargs)
        elif provider == "vllm":
            return VLLMInferenceEngine(**engine_kwargs)
        elif provider == "sglang":
            return SGLangInferenceEngine(**engine_kwargs)
        elif provider == "openrouter":
            return OpenRouterInferenceEngine(**engine_kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    async def list_sessions(self):
        return self.db.list_sessions()

    async def get_indexer_job_manager(self, session_id: str) -> Optional["JobManager"]:
        """Return the document indexer JobManager for a session, loading it if needed."""
        async with self._lock:
            if session_id in self._cache:
                return self._cache[session_id].get("indexer_job_manager")
        # Not cached — load from storage (this also caches it)
        await self._load_session_from_storage(session_id)
        async with self._lock:
            if session_id in self._cache:
                return self._cache[session_id].get("indexer_job_manager")
        return None

    async def delete_session(self, session_id: str):
        async with self._lock:
            cache_entry = self._cache.pop(session_id, None)
            if cache_entry:
                for key in ("worker", "indexer_worker"):
                    w = cache_entry.get(key)
                    if w:
                        try:
                            w.shutdown()
                        except Exception:
                            pass
        self.db.delete_session(session_id)
        session_path = os.path.join(self.sessions_dir, session_id)
        if os.path.exists(session_path):
            shutil.rmtree(session_path)
        return True
    
    async def rename_session(self, session_id: str, new_name: str):
        self.db.rename_session(session_id, new_name)
        return True