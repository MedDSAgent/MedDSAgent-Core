import type { InternalDatabase } from "../db/index.js";
import type { AgentEvent } from "../agents/index.js";
import type { Step } from "../history/index.js";

// ---------------------------------------------------------------------------
// Session config — mirrors Python SessionConfig (all optional; defaults in SM)
// ---------------------------------------------------------------------------

export interface SessionConfig {
  llm_provider?: string;
  llm_model?: string;
  llm_api_key?: string;
  llm_base_url?: string;
  temperature?: number;
  top_p?: number;
  llm_api_version?: string;
  db_connection_code?: string;
  language?: string;
  reasoning_effort?: string;
  specialty_id?: string;
  specialty_prompt?: string;
  python_bin?: string;
  r_home?: string;
  [key: string]: unknown;
}

export interface SessionSummary {
  session_id: string;
  name: string;
  last_accessed: string;
}

export interface SessionDetail {
  session_id: string;
  name: string;
  last_accessed: string;
  config: SessionConfig;
}

export interface VariablesSnapshot {
  python?: unknown[];
  r?: unknown[];
  language?: string;
}

// ---------------------------------------------------------------------------
// ISessionManager — the interface the Fastify server depends on.
// Phase 6 (SessionManager.ts) implements this.
// ---------------------------------------------------------------------------

export interface ISessionManager {
  readonly sessionsDir: string;
  readonly db: InternalDatabase;

  // --- Session CRUD ---
  listSessions(): Promise<SessionSummary[]>;
  createSession(name: string, config: SessionConfig): Promise<string>;
  getSession(sessionId: string): Promise<SessionDetail | null>;
  updateSession(sessionId: string, name: string, config: SessionConfig): Promise<void>;
  deleteSession(sessionId: string): Promise<void>;
  renameSession(sessionId: string, name: string): Promise<void>;

  // --- Chat ---
  getHistory(sessionId: string): Step[];
  chat(sessionId: string, message: string): AsyncGenerator<AgentEvent>;
  stopSession(sessionId: string): Promise<boolean>;
  addSystemMessage(sessionId: string, message: string): Promise<void>;

  // --- State inspection ---
  getVariables(sessionId: string): Promise<VariablesSnapshot | null>;
  getMemoryDebug(sessionId: string): Promise<Record<string, unknown> | null>;
  getCompressionDebug(sessionId: string): Promise<Record<string, unknown> | null>;
}
