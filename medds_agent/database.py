import sqlite3
import json
import os
import importlib.resources
from datetime import datetime
from typing import List, Dict, Any, Optional
from medds_agent.history import Step, SystemStep, UserStep, AgentStep, ObservationStep


class InternalDatabase:
    def __init__(self, db_path: str):
        """
        Initialize the internal database connection.
        
        Parameters:
        -----------
        db_path : str
            The full path to the runtime .db file (e.g., /work/internal.db).
        """
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a configured database connection."""
        conn = sqlite3.connect(self.db_path)
        # Enable accessing columns by name
        conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        """Initialize the database schema from medds_agent/asset/database_queries.sql."""
        # Ensure the directory for the DB file exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        schema = ""
        try:
            # Load from the asset package
            ref = importlib.resources.files('medds_agent.asset').joinpath('database_queries.sql')
            with importlib.resources.as_file(ref) as sql_path:
                with open(sql_path, 'r', encoding='utf-8') as f:
                    schema = f.read()
        except Exception as e:
            # Fallback for local development
            current_dir = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(current_dir, 'asset', 'database_queries.sql')
            
            if os.path.exists(local_path):
                with open(local_path, 'r', encoding='utf-8') as f:
                    schema = f.read()
            else:
                raise FileNotFoundError(f"Could not find database_queries.sql. Error: {e}")

        if schema:
            with self._get_conn() as conn:
                conn.executescript(schema)
                conn.commit()
                
    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(self, session_id: str, name: str, work_dir: str, config: Optional[Dict[str, Any]] = None):
        """Create a new session entry."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, name, config, work_dir, last_accessed) VALUES (?, ?, ?, ?, ?)",
                (session_id, name, json.dumps(config) if config else None, work_dir, datetime.now())
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session metadata."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            return dict(row) if row else None

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List sessions ordered by creation time (newest first)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY rowid DESC LIMIT ?", 
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def update_last_accessed(self, session_id: str):
        """Update the last_accessed timestamp for a session."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                (datetime.now(), session_id)
            )
            conn.commit()

    def delete_session(self, session_id: str):
        """Delete a session and all associated history/state (via cascade)."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            
    def rename_session(self, session_id: str, new_name: str):
        """Rename an existing session."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET name = ? WHERE session_id = ?",
                (new_name, session_id)
            )
            conn.commit()

    def save_session_config(self, session_id: str, config: Dict[str, Any]):
        """Save session configuration to the database."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET config = ? WHERE session_id = ?",
                (json.dumps(config), session_id)
            )
            conn.commit()

    def get_session_config(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session configuration from the database."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT config, specialty_id, specialty_prompt FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if not row or not row['config']:
                return None
            config = json.loads(row['config'])
            # Merge specialty from dedicated columns into config
            if row['specialty_id']:
                config['specialty_id'] = row['specialty_id']
            if row['specialty_prompt']:
                config['specialty_prompt'] = row['specialty_prompt']
            return config

    def save_session_specialty(self, session_id: str, specialty_id: Optional[str], specialty_prompt: Optional[str]):
        """Save specialty ID and prompt to dedicated columns."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET specialty_id = ?, specialty_prompt = ? WHERE session_id = ?",
                (specialty_id, specialty_prompt, session_id)
            )
            conn.commit()

    # =========================================================================
    # History Management
    # =========================================================================

    def add_history_step(self, session_id: str, round_num: int, step: Step):
        """
        Record a single conversation step object.
        
        Parameters:
        -----------
        session_id : str
            The session identifier.
        round_num : int
            The round number this step belongs to.
        step : Step
            The step object to serialize and store.
        """
        # Serialize the step to get data and type
        step_data = step.serialize()
        step_type = step_data.get("type", step.__class__.__name__)
        
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO session_history_steps 
                (session_id, round_num, step_type, step_data, created_at) 
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, round_num, step_type, json.dumps(step_data), datetime.now())
            )
            conn.commit()

    def get_history_steps(self, session_id: str) -> List[Step]:
        """
        Retrieve all history steps for a session as Step objects.
        
        Returns:
        --------
        List[Step]
            A list of deserialized Step objects, ordered by creation time.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM session_history_steps WHERE session_id = ? ORDER BY step_id ASC",
                (session_id,)
            ).fetchall()
            
            steps = []
            for row in rows:
                item = dict(row)
                step_type = item['step_type']
                step_data = json.loads(item['step_data']) if item.get('step_data') else {}
                
                # Deserialization Factory Logic
                try:
                    if step_type == "SystemStep":
                        step = SystemStep.deserialize(step_data)
                    elif step_type == "UserStep":
                        step = UserStep.deserialize(step_data)
                    elif step_type == "AgentStep":
                        step = AgentStep.deserialize(step_data)
                    elif step_type == "ObservationStep":
                        step = ObservationStep.deserialize(step_data)
                    else:
                        # Fallback for unknown types if necessary, or skip
                        print(f"Warning: Unknown step type '{step_type}' in DB for session {session_id}")
                        continue
                        
                    steps.append(step)
                except Exception as e:
                    print(f"Error deserializing step {item.get('step_id')}: {e}")
                    
            return steps

    # =========================================================================
    # State Management
    # =========================================================================

    def save_session_state(self, session_id: str, agent_memory_data: Optional[Dict], 
                           python_state_path: str = None, r_state_path: str = None):
        """
        Upsert (Insert or Replace) the session state configuration.
        """
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO session_states (session_id, agent_memory_data, python_state_path, r_state_path, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    agent_memory_data = excluded.agent_memory_data,
                    python_state_path = excluded.python_state_path,
                    r_state_path = excluded.r_state_path,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id, 
                    json.dumps(agent_memory_data) if agent_memory_data else None, 
                    python_state_path, 
                    r_state_path, 
                    datetime.now()
                )
            )
            conn.commit()

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the saved state paths and memory configuration."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM session_states WHERE session_id = ?", (session_id,)).fetchone()
            if not row:
                return None

            data = dict(row)
            if data.get('agent_memory_data'):
                data['agent_memory_data'] = json.loads(data['agent_memory_data'])
            return data

    # =========================================================================
    # Parsed Document Management
    # =========================================================================

    def get_parsed_document(self, session_id: str, file_name: str) -> Optional[Dict[str, Any]]:
        """Get parsed document record by session and filename."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM parsed_documents WHERE session_id = ? AND file_name = ?",
                (session_id, file_name)
            ).fetchone()
            return dict(row) if row else None

    def list_parsed_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """List all parsed documents for a session."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM parsed_documents WHERE session_id = ? ORDER BY parsed_at DESC",
                (session_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def upsert_parsed_document(self, session_id: str, file_name: str, file_hash: str) -> int:
        """
        Insert or replace a parsed document record.
        Deletes old sections on conflict (via CASCADE from delete + re-insert).

        Returns the document_id.
        """
        with self._get_conn() as conn:
            # Delete existing record (cascades to sections)
            conn.execute(
                "DELETE FROM parsed_documents WHERE session_id = ? AND file_name = ?",
                (session_id, file_name)
            )
            cursor = conn.execute(
                "INSERT INTO parsed_documents (session_id, file_name, file_hash) VALUES (?, ?, ?)",
                (session_id, file_name, file_hash)
            )
            conn.commit()
            return cursor.lastrowid

    def insert_sections(self, document_id: int, sections: List[Dict[str, Any]]):
        """
        Bulk insert sections for a parsed document.

        Each section dict should have: section_id, parent_section_id, title, level, content, section_order
        """
        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO document_sections
                (document_id, section_id, parent_section_id, title, level, content, section_order)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        s["section_id"],
                        s.get("parent_section_id"),
                        s["title"],
                        s["level"],
                        s["content"],
                        s["section_order"]
                    )
                    for s in sections
                ]
            )
            conn.commit()

    def get_document_sections(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all sections for a document, ordered by section_order."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM document_sections WHERE document_id = ? ORDER BY section_order ASC",
                (document_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_section_by_id(self, document_id: int, section_id: str) -> Optional[Dict[str, Any]]:
        """Get a single section by document_id and section_id."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM document_sections WHERE document_id = ? AND section_id = ?",
                (document_id, section_id)
            ).fetchone()
            return dict(row) if row else None

    def get_child_sections(self, document_id: int, parent_section_id: str) -> List[Dict[str, Any]]:
        """Get direct child sections of a given parent."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM document_sections WHERE document_id = ? AND parent_section_id = ? ORDER BY section_order ASC",
                (document_id, parent_section_id)
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_parsed_document(self, session_id: str, file_name: str):
        """Delete a parsed document and its sections (via CASCADE)."""
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM parsed_documents WHERE session_id = ? AND file_name = ?",
                (session_id, file_name)
            )
            conn.commit()