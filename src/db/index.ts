import { DatabaseSync } from "node:sqlite";
import { readFileSync } from "fs";
import { mkdirSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import type { Step } from "../history/index.js";
import { serializeStep, deserializeStep } from "../history/index.js";

// ---------------------------------------------------------------------------
// Schema path — resolve relative to this file, works for both dev and dist
// ---------------------------------------------------------------------------

const __dirname = dirname(fileURLToPath(import.meta.url));

function resolveSchemaPath(): string {
  return join(__dirname, "schema.sql");
}

// ---------------------------------------------------------------------------
// Row types
// ---------------------------------------------------------------------------

export interface SessionRow {
  session_id: string;
  name: string;
  config: string | null;
  specialty_id: string | null;
  specialty_prompt: string | null;
  created_at: string;
  last_accessed: string;
  work_dir: string;
}

export interface SessionStateRow {
  session_id: string;
  agent_memory_data: string | null;
  python_state_path: string | null;
  r_state_path: string | null;
  updated_at: string;
}

export interface ParsedDocumentRow {
  document_id: number;
  session_id: string;
  file_name: string;
  file_hash: string;
  status: string;
  error_message: string | null;
  parsed_at: string;
}

// ---------------------------------------------------------------------------
// InternalDatabase
// ---------------------------------------------------------------------------

export class InternalDatabase {
  private readonly db: DatabaseSync;

  constructor(dbPath: string) {
    mkdirSync(dirname(dbPath), { recursive: true });
    this.db = new DatabaseSync(dbPath, { enableForeignKeyConstraints: true });
    this.db.exec("PRAGMA journal_mode = WAL");
    this._initSchema();
    this._migrate();
  }

  close(): void {
    this.db.close();
  }

  private _initSchema(): void {
    const schema = readFileSync(resolveSchemaPath(), "utf-8");
    this.db.exec(schema);
  }

  private _migrate(): void {
    for (const [col, def] of [
      ["status", "TEXT NOT NULL DEFAULT 'done'"],
      ["error_message", "TEXT"],
    ] as [string, string][]) {
      try {
        this.db.exec(`ALTER TABLE parsed_documents ADD COLUMN ${col} ${def}`);
      } catch {
        // Column already exists — expected on existing DBs
      }
    }
  }

  // =========================================================================
  // Sessions
  // =========================================================================

  createSession(sessionId: string, name: string, workDir: string, config?: Record<string, unknown>): void {
    this.db
      .prepare(
        "INSERT INTO sessions (session_id, name, config, work_dir, last_accessed) VALUES (?, ?, ?, ?, ?)",
      )
      .run(sessionId, name, config ? JSON.stringify(config) : null, workDir, new Date().toISOString());
  }

  getSession(sessionId: string): SessionRow | undefined {
    return this.db
      .prepare("SELECT * FROM sessions WHERE session_id = ?")
      .get(sessionId) as SessionRow | undefined;
  }

  listSessions(limit = 50): SessionRow[] {
    return this.db
      .prepare("SELECT * FROM sessions ORDER BY rowid DESC LIMIT ?")
      .all(limit) as unknown as SessionRow[];
  }

  updateLastAccessed(sessionId: string): void {
    this.db
      .prepare("UPDATE sessions SET last_accessed = ? WHERE session_id = ?")
      .run(new Date().toISOString(), sessionId);
  }

  deleteSession(sessionId: string): void {
    this.db.prepare("DELETE FROM sessions WHERE session_id = ?").run(sessionId);
  }

  renameSession(sessionId: string, newName: string): void {
    this.db.prepare("UPDATE sessions SET name = ? WHERE session_id = ?").run(newName, sessionId);
  }

  saveSessionConfig(sessionId: string, config: Record<string, unknown>): void {
    this.db
      .prepare("UPDATE sessions SET config = ? WHERE session_id = ?")
      .run(JSON.stringify(config), sessionId);
  }

  getSessionConfig(sessionId: string): Record<string, unknown> | null {
    const row = this.db
      .prepare("SELECT config, specialty_id, specialty_prompt FROM sessions WHERE session_id = ?")
      .get(sessionId) as { config: string | null; specialty_id: string | null; specialty_prompt: string | null } | undefined;

    if (!row?.config) return null;
    const cfg = JSON.parse(row.config) as Record<string, unknown>;
    if (row.specialty_id) cfg["specialty_id"] = row.specialty_id;
    if (row.specialty_prompt) cfg["specialty_prompt"] = row.specialty_prompt;
    return cfg;
  }

  saveSessionSpecialty(sessionId: string, specialtyId: string | null, specialtyPrompt: string | null): void {
    this.db
      .prepare("UPDATE sessions SET specialty_id = ?, specialty_prompt = ? WHERE session_id = ?")
      .run(specialtyId, specialtyPrompt, sessionId);
  }

  // =========================================================================
  // History
  // =========================================================================

  addHistoryStep(sessionId: string, roundNum: number, step: Step): void {
    const raw = serializeStep(step);
    this.db
      .prepare(
        `INSERT INTO session_history_steps
         (session_id, round_num, step_type, step_data, created_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run(sessionId, roundNum, step.type, JSON.stringify(raw), new Date().toISOString());
  }

  getHistorySteps(sessionId: string): Step[] {
    const rows = this.db
      .prepare(
        "SELECT * FROM session_history_steps WHERE session_id = ? ORDER BY step_id ASC",
      )
      .all(sessionId) as { step_type: string; step_data: string }[];

    const steps: Step[] = [];
    for (const row of rows) {
      try {
        const raw = JSON.parse(row.step_data) as Record<string, unknown>;
        steps.push(deserializeStep(raw));
      } catch (e) {
        console.warn(`Failed to deserialize step for session ${sessionId}:`, e);
      }
    }
    return steps;
  }

  // =========================================================================
  // Session State
  // =========================================================================

  saveSessionState(
    sessionId: string,
    agentMemoryData: Record<string, unknown> | null,
    pythonStatePath?: string | null,
    rStatePath?: string | null,
  ): void {
    this.db
      .prepare(
        `INSERT INTO session_states (session_id, agent_memory_data, python_state_path, r_state_path, updated_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(session_id) DO UPDATE SET
           agent_memory_data = excluded.agent_memory_data,
           python_state_path = excluded.python_state_path,
           r_state_path = excluded.r_state_path,
           updated_at = excluded.updated_at`,
      )
      .run(
        sessionId,
        agentMemoryData ? JSON.stringify(agentMemoryData) : null,
        pythonStatePath ?? null,
        rStatePath ?? null,
        new Date().toISOString(),
      );
  }

  getSessionState(sessionId: string): SessionStateRow | null {
    const row = this.db
      .prepare("SELECT * FROM session_states WHERE session_id = ?")
      .get(sessionId) as SessionStateRow | undefined;
    return row ?? null;
  }

  // =========================================================================
  // Parsed Documents
  // =========================================================================

  getParsedDocument(sessionId: string, fileName: string): ParsedDocumentRow | null {
    return (
      (this.db
        .prepare("SELECT * FROM parsed_documents WHERE session_id = ? AND file_name = ?")
        .get(sessionId, fileName) as ParsedDocumentRow | undefined) ?? null
    );
  }

  listParsedDocuments(sessionId: string): ParsedDocumentRow[] {
    return this.db
      .prepare("SELECT * FROM parsed_documents WHERE session_id = ? ORDER BY parsed_at DESC")
      .all(sessionId) as unknown as ParsedDocumentRow[];
  }

  markIndexingStarted(sessionId: string, fileName: string, fileHash: string): void {
    this.db
      .prepare(
        `INSERT INTO parsed_documents (session_id, file_name, file_hash, status)
         VALUES (?, ?, ?, 'indexing')
         ON CONFLICT(session_id, file_name) DO UPDATE SET
           file_hash = excluded.file_hash,
           status = 'indexing',
           error_message = NULL,
           parsed_at = CURRENT_TIMESTAMP`,
      )
      .run(sessionId, fileName, fileHash);
  }

  markIndexingFailed(sessionId: string, fileName: string, errorMessage: string): void {
    this.db
      .prepare(
        "UPDATE parsed_documents SET status = 'failed', error_message = ? WHERE session_id = ? AND file_name = ?",
      )
      .run(errorMessage, sessionId, fileName);
  }

  upsertParsedDocument(sessionId: string, fileName: string, fileHash: string): number {
    this.db
      .prepare("DELETE FROM parsed_documents WHERE session_id = ? AND file_name = ?")
      .run(sessionId, fileName);
    const result = this.db
      .prepare("INSERT INTO parsed_documents (session_id, file_name, file_hash) VALUES (?, ?, ?)")
      .run(sessionId, fileName, fileHash);
    return Number(result.lastInsertRowid);
  }

  insertSections(
    documentId: number,
    sections: Array<{
      section_id: string;
      parent_section_id?: string | null;
      title: string;
      level: number;
      content: string;
      section_order: number;
    }>,
  ): void {
    const stmt = this.db.prepare(
      `INSERT INTO document_sections
       (document_id, section_id, parent_section_id, title, level, content, section_order)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    );
    this.db.exec("BEGIN");
    try {
      for (const s of sections) {
        stmt.run(documentId, s.section_id, s.parent_section_id ?? null, s.title, s.level, s.content, s.section_order);
      }
      this.db.exec("COMMIT");
    } catch (e) {
      this.db.exec("ROLLBACK");
      throw e;
    }
  }

  getDocumentSections(documentId: number): Record<string, unknown>[] {
    return this.db
      .prepare("SELECT * FROM document_sections WHERE document_id = ? ORDER BY section_order ASC")
      .all(documentId) as Record<string, unknown>[];
  }

  deleteParsedDocument(sessionId: string, fileName: string): void {
    this.db
      .prepare("DELETE FROM parsed_documents WHERE session_id = ? AND file_name = ?")
      .run(sessionId, fileName);
  }
}
