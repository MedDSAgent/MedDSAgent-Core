-- =============================================================================
-- 1. Sessions Table
-- Stores metadata about the session workspace.
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config JSON,
    specialty_id TEXT,
    specialty_prompt TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    work_dir TEXT NOT NULL
);

-- =============================================================================
-- 2. Session History Steps Table
-- Stores every atomic step in the conversation (User, Tool Call, Output, Answer).
-- Granularity: One row per step.
-- =============================================================================
CREATE TABLE IF NOT EXISTS session_history_steps (
    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    step_type TEXT NOT NULL,     -- 'UserStep', 'ToolCallStep', 'ToolOutputStep', 'AnswerStep'
    step_data JSON NOT NULL,     -- The serialized step object (content, args, timestamp)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- =============================================================================
-- 3. Session States Table
-- Stores the checkpoint for the agent's memory and execution environments.
-- One row per session (1:1 relationship).
-- =============================================================================
CREATE TABLE IF NOT EXISTS session_states (
    session_id TEXT PRIMARY KEY,
    agent_memory_data JSON,
    python_state_path TEXT,      -- e.g., "sessions/{id}/state.pkl"
    r_state_path TEXT,           -- e.g., "sessions/{id}/.RData"
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- =============================================================================
-- 4. Parsed Documents Table
-- Tracks documents that have been parsed and indexed for search.
-- =============================================================================
CREATE TABLE IF NOT EXISTS parsed_documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    UNIQUE(session_id, file_name)
);

-- =============================================================================
-- 5. Document Sections Table
-- Stores hierarchical sections extracted from parsed documents.
-- =============================================================================
CREATE TABLE IF NOT EXISTS document_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    section_id TEXT NOT NULL,
    parent_section_id TEXT,
    title TEXT NOT NULL,
    level INTEGER NOT NULL,
    content TEXT NOT NULL,
    section_order INTEGER NOT NULL,
    FOREIGN KEY (document_id) REFERENCES parsed_documents(document_id) ON DELETE CASCADE
);

-- =============================================================================
-- 6. Indexes
-- Optimize lookup performance.
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_history_session_id ON session_history_steps(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_accessed ON sessions(last_accessed);
CREATE INDEX IF NOT EXISTS idx_parsed_docs_session ON parsed_documents(session_id);
CREATE INDEX IF NOT EXISTS idx_doc_sections_document ON document_sections(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_sections_section_id ON document_sections(document_id, section_id);