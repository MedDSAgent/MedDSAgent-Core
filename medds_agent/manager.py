import os
import uuid
import asyncio
import shutil
import dill
import json
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

from medds_agent.agents import Agent
from medds_agent.history import History, SystemStep
from medds_agent.memory import SlideWindowAgentMemory, IndexedAgentMemory
from medds_agent.database import InternalDatabase
from medds_agent.tools import PythonExecutorTool, RExecutorTool, FileSystemTool, DocumentSearchTool

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
        
        # In-memory cache: {session_id: {"agent": Agent, "last_active": datetime, "persisted_steps": int}}
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
            agent, connection_info = self._init_agent_instance(session_dir, config, session_id=session_id)
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

        # 5. Cache it
        async with self._lock:
            self._cache[session_id] = {
                "agent": agent,
                "last_active": datetime.now(),
                "persisted_steps": 0
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
        """Update session name and configuration, reinitializing DB connection if needed."""
        # 1. Update Name in DB
        self.db.rename_session(session_id, name)

        # 2. Get old config to compare DB connection code
        old_config = self.db.get_session_config(session_id) or {}
        old_db_code = (old_config.get("db_connection_code") or "").strip()
        new_db_code = (config.get("db_connection_code") or "").strip()

        # 3. Update Config in DB
        self.db.save_session_config(session_id, config)

        # 3b. Save specialty to dedicated columns
        self.db.save_session_specialty(
            session_id,
            specialty_id=config.get("specialty_id"),
            specialty_prompt=config.get("specialty_prompt")
        )

        # 4. Invalidate Cache to force reload with new settings
        async with self._lock:
            if session_id in self._cache:
                del self._cache[session_id]

        # 5. If DB connection code changed (added or modified), execute it and add system step
        if new_db_code and new_db_code != old_db_code:
            # Load the agent (this will reinitialize with new config)
            agent = await self._load_session_from_storage(session_id)
            if agent:
                language = config.get("language", "python").lower()

                if language == "python":
                    # Find the python tool and get connection info
                    python_tool = next((t for t in agent.tools if isinstance(t, PythonExecutorTool)), None)
                    if python_tool:
                        db_engine = python_tool._globals.get('db_engine')
                        conn = python_tool._globals.get('conn')

                        if db_engine is not None or conn is not None:
                            connection_info = self._get_connection_info(python_tool)
                            if connection_info:
                                await self.add_system_message(
                                    session_id,
                                    f"Database connection established. Use variable '{connection_info['variable']}' ({connection_info['type']}) to query the database."
                                )

                elif language == "r":
                    r_tool = next((t for t in agent.tools if isinstance(t, RExecutorTool)), None)
                    if r_tool:
                        connection_info = self._get_r_connection_info(r_tool)
                        if connection_info:
                            await self.add_system_message(
                                session_id,
                                f"Database connection established. Use variable '{connection_info['variable']}' ({connection_info['type']}) to query the database."
                            )

    def _get_connection_info(self, python_tool: PythonExecutorTool) -> Optional[Dict[str, str]]:
        """Get connection info from PythonExecutorTool's globals."""
        db_engine = python_tool._globals.get('db_engine')
        conn = python_tool._globals.get('conn')

        connection_info = None

        if db_engine is not None:
            engine_type = type(db_engine).__name__
            try:
                dialect = getattr(db_engine, 'dialect', None)
                if dialect:
                    dialect_name = getattr(dialect, 'name', 'unknown')
                    engine_type = f"SQLAlchemy ({dialect_name})"
                else:
                    engine_type = "SQLAlchemy Engine"
            except:
                engine_type = "SQLAlchemy Engine"
            connection_info = {"variable": "db_engine", "type": engine_type}

        if conn is not None:
            conn_type = type(conn).__module__ + "." + type(conn).__name__
            if "oracledb" in conn_type.lower():
                conn_type = "Oracle (oracledb)"
            elif "psycopg" in conn_type.lower():
                conn_type = "PostgreSQL (psycopg)"
            elif "mysql" in conn_type.lower():
                conn_type = "MySQL"
            elif "sqlite" in conn_type.lower():
                conn_type = "SQLite"
            elif "pyodbc" in conn_type.lower():
                conn_type = "ODBC"
            else:
                conn_type = type(conn).__name__

            if connection_info:
                connection_info = {"variable": "db_engine, conn", "type": f"{connection_info['type']} + {conn_type}"}
            else:
                connection_info = {"variable": "conn", "type": conn_type}

        return connection_info

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

            # 2. Save Executor State (Python or R)
            session_dir = os.path.join(self.sessions_dir, session_id)
            python_tool = next((t for t in agent.tools if isinstance(t, PythonExecutorTool)), None)
            r_tool = next((t for t in agent.tools if isinstance(t, RExecutorTool)), None)

            if python_tool and hasattr(python_tool, '_globals'):
                state_path = os.path.join(session_dir, "state.pkl")
                try:
                    # Filter out connection objects that can't be pickled
                    # They will be recreated on session reload via db_connection_code
                    state_to_save = {
                        k: v for k, v in python_tool._globals.items()
                        if k not in ('db_engine', 'conn')
                    }
                    with open(state_path, 'wb') as f:
                        dill.dump(state_to_save, f)

                    self.db.save_session_state(
                        session_id=session_id,
                        agent_memory_data=agent_memory_data,
                        python_state_path=f"sessions/{session_id}/state.pkl"
                    )
                except Exception as e:
                    print(f"Error saving python state for {session_id}: {e}")

            elif r_tool and hasattr(r_tool, 'save_state'):
                r_state_path = os.path.join(session_dir, "state.RData")
                try:
                    r_tool.save_state(r_state_path)
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

    def _execute_db_connection_code(self, python_tool: PythonExecutorTool, code: str) -> Optional[Dict[str, str]]:
        """
        Execute database connection code and inject the resulting objects into PythonExecutorTool state.
        The code should create either 'db_engine' (SQLAlchemy) or 'conn' (direct connection) variable.

        Returns:
            Dict with connection info if successful, None if failed.
            Example: {"variable": "db_engine", "type": "SQLAlchemy Engine"}
        """
        try:
            # Execute in the PythonExecutorTool's namespace so it persists
            exec(code, python_tool._globals, python_tool._locals)
            python_tool._globals.update(python_tool._locals)

            # Get connection info using helper
            connection_info = self._get_connection_info(python_tool)

            if connection_info:
                print(f"Database connection established: {connection_info['variable']} ({connection_info['type']})")
            else:
                print("Warning: DB connection code executed but no 'db_engine' or 'conn' created")

            return connection_info

        except Exception as e:
            print(f"Error executing DB connection code: {type(e).__name__}: {str(e)}")
            # Don't raise - let the session continue without DB connection
            return None

    def _execute_r_db_connection_code(self, r_tool: RExecutorTool, code: str) -> Optional[Dict[str, str]]:
        """
        Execute R database connection code in the RExecutorTool environment.
        The code should create a connection variable (e.g. 'con') using DBI/odbc/RPostgres etc.

        Returns:
            Dict with connection info if successful, None if failed.
            Example: {"variable": "con", "type": "PqConnection (RPostgres)"}
        """
        try:
            output = r_tool.execute({"code": code})

            if "[Error]" in output:
                print(f"Error executing R DB connection code: {output}")
                return None

            # Check the R environment for known connection classes
            connection_info = self._get_r_connection_info(r_tool)

            if connection_info:
                print(f"R database connection established: {connection_info['variable']} ({connection_info['type']})")
            else:
                print("Warning: R DB connection code executed but no connection object found")

            return connection_info

        except Exception as e:
            print(f"Error executing R DB connection code: {type(e).__name__}: {str(e)}")
            return None

    def _get_r_connection_info(self, r_tool: RExecutorTool) -> Optional[Dict[str, str]]:
        """Get connection info from RExecutorTool's environment."""
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import pandas2ri
        import rpy2.robjects as robjects

        with localconverter(robjects.default_converter + pandas2ri.converter):
            obj_names = list(r_tool.env.keys())

            for name in obj_names:
                obj = r_tool.env[name]
                try:
                    r_class_list = list(obj.rclass)
                except:
                    continue

                # Check for DBI connection classes
                dbi_classes = [
                    "PqConnection", "MariaDBConnection", "MySQLConnection",
                    "OraConnection", "Microsoft SQL Server",
                    "SQLiteConnection", "OdbcConnection", "DBIConnection"
                ]
                for cls in r_class_list:
                    if any(dbi_cls in cls for dbi_cls in dbi_classes) or "Connection" in cls:
                        return {"variable": name, "type": cls}

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
            agent, _ = self._init_agent_instance(session_dir, config, session_id=session_id)
        except Exception as e:
            print(f"Error initializing agent for {session_id}: {e}")
            return None
        
        # 3. Reconstruct History
        history_steps = self.db.get_history_steps(session_id)
        history = History()
        for step in history_steps:
            history.add_step(step) 
        agent.history = history
        
        # 4. Load Executor State (Python or R)
        python_tool = next((t for t in agent.tools if isinstance(t, PythonExecutorTool)), None)
        r_tool = next((t for t in agent.tools if isinstance(t, RExecutorTool)), None)

        if python_tool:
            state_path = os.path.join(session_dir, "state.pkl")
            if os.path.exists(state_path):
                try:
                    with open(state_path, 'rb') as f:
                        python_tool._globals.update(dill.load(f))
                except Exception as e:
                    print(f"Warning: Failed to load Python state for {session_id}: {e}")

        elif r_tool:
            r_state_path = os.path.join(session_dir, "state.RData")
            if os.path.exists(r_state_path):
                try:
                    r_tool.load_state(r_state_path)
                except Exception as e:
                    print(f"Warning: Failed to load R state for {session_id}: {e}")

        # 5. Restore Agent Memory State
        session_state = self.db.get_session_state(session_id)
        if session_state and session_state.get('agent_memory_data'):
            try:
                agent.agent_memory.deserialize(session_state['agent_memory_data'])
            except Exception as e:
                print(f"Warning: Failed to restore agent memory for {session_id}: {e}")

        # 6. Cache
        total_steps = sum(len(r.steps) for r in history.rounds)
        async with self._lock:
            self._cache[session_id] = {
                "agent": agent,
                "last_active": datetime.now(),
                "persisted_steps": total_steps
            }
            
        return agent

    def _init_agent_instance(self, session_dir: str, config: Dict[str, Any], session_id: str = None) -> Tuple[Agent, Optional[Dict[str, str]]]:
        """
        Construct Agent based on dynamic configuration.

        Returns:
            Tuple of (Agent, connection_info) where connection_info is None if no DB connection.
        """

        # 1. Configure Tools based on language
        language = config.get("language", "python").lower()
        connection_info = None

        if language == "r":
            executor_tool = RExecutorTool(work_dir=session_dir)
        else:
            executor_tool = PythonExecutorTool(work_dir=session_dir)

        tools = [
            executor_tool,
            FileSystemTool(dir=session_dir),
        ]

        # Add DocumentSearchTool if session_id is available
        if session_id:
            tools.append(DocumentSearchTool(session_id=session_id, db=self.db))

        # Execute DB connection code if provided (Python-only)
        if language == "python" and isinstance(executor_tool, PythonExecutorTool):
            db_connection_code = config.get("db_connection_code")
            if db_connection_code and db_connection_code.strip():
                connection_info = self._execute_db_connection_code(executor_tool, db_connection_code)

        # Execute DB connection code if provided (R)
        elif language == "r" and isinstance(executor_tool, RExecutorTool):
            db_connection_code = config.get("db_connection_code")
            if db_connection_code and db_connection_code.strip():
                connection_info = self._execute_r_db_connection_code(executor_tool, db_connection_code)
            
        # 2. Configure LLM Engine
        provider = config.get("llm_provider", "openai").lower()
        model = config.get("llm_model", "gpt-4")
        api_key = config.get("llm_api_key")
        base_url = config.get("llm_base_url")
        
        # Get sampling parameters from config
        temperature = float(config.get("temperature", 1.0))
        top_p = float(config.get("top_p", 1.0))
        
        # Get max_new_tokens from env variable
        max_new_tokens = int(os.environ.get("LLM_MAX_NEW_TOKENS", "16384"))
        max_input_tokens = int(os.environ.get("LLM_MAX_INPUT_TOKENS", "120000"))
        
        # Select LLM config based on reasoning effort
        reasoning_effort = config.get("reasoning_effort")
        if reasoning_effort and reasoning_effort in ["low", "medium", "high"]:
            llm_config = AgentReasoningLLMConfig(
                max_input_tokens=max_input_tokens
            )
        else:
            llm_config = AgentBasicLLMConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                max_input_tokens=max_input_tokens
            )
        
        llm_engine = None
        
        # Base Arguments for all engines
        engine_kwargs = {
            "model": model, 
            "config": llm_config
        }
        
        # Only add optional arguments if they are not None
        if api_key:
            engine_kwargs["api_key"] = api_key
            # For OpenAI, also set env var if provided
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            if provider == "azure":
                os.environ["AZURE_OPENAI_API_KEY"] = api_key

        if base_url:
            if provider == "azure":
                engine_kwargs["azure_endpoint"] = base_url
            else:
                engine_kwargs["base_url"] = base_url
        
        # Provider Dispatch
        if provider == "openai":
            llm_engine = OpenAIInferenceEngine(**engine_kwargs)
            
        elif provider == "azure":
            # Azure needs api_version specifically
            if config.get("llm_api_version"):
                engine_kwargs["api_version"] = config.get("llm_api_version")
            else:
                engine_kwargs["api_version"] = "2023-12-01-preview" # Fallback if not in config
                
            llm_engine = AzureOpenAIInferenceEngine(**engine_kwargs)
            
        elif provider == "vllm":
            llm_engine = VLLMInferenceEngine(**engine_kwargs)
            
        elif provider == "sglang":
            llm_engine = SGLangInferenceEngine(**engine_kwargs)
            
        elif provider == "openrouter":
            llm_engine = OpenRouterInferenceEngine(**engine_kwargs)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

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

        # 4. Create Agent
        agent = Agent(
            llm_engine=llm_engine,
            history=History(),
            agent_memory=agent_memory,
            specialty_prompt=config.get("specialty_prompt", None),
            tools=tools,
            verbose=False
        )
        return agent, connection_info

    async def list_sessions(self):
        return self.db.list_sessions()

    async def delete_session(self, session_id: str):
        async with self._lock:
            self._cache.pop(session_id, None)
        self.db.delete_session(session_id)
        session_path = os.path.join(self.sessions_dir, session_id)
        if os.path.exists(session_path):
            shutil.rmtree(session_path)
        return True
    
    async def rename_session(self, session_id: str, new_name: str):
        self.db.rename_session(session_id, new_name)
        return True