import * as fs from "fs";
import * as path from "path";
import { randomUUID } from "crypto";
import { fileURLToPath } from "url";

// Root of the MedDSAgent-Core package — python_worker/ lives here.
// dist/session/index.js → dist/ → core root (two levels up)
const _CORE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
import { InternalDatabase } from "../db/index.js";
import { Agent } from "../agents/index.js";
import type { AgentEvent } from "../agents/index.js";
import { IndexedAgentMemory } from "../memory/index.js";
import { makeHistory, makeSystemStep, addStep } from "../history/index.js";
import type { HistoryData, Step } from "../history/index.js";
import { createEngine } from "../engines/index.js";
import type { LLMEngine } from "../engines/index.js";
import { WorkerProcess } from "../workers/WorkerProcess.js";
import { JobManager } from "../jobs/index.js";
import { PythonExecutorTool, RExecutorTool } from "../tools/python.js";
import { FileSystemTool } from "../tools/filesystem.js";
import type { ISessionManager, SessionConfig, SessionSummary, SessionDetail, VariablesSnapshot } from "../server/types.js";

// ---------------------------------------------------------------------------
// In-memory cache entry
// ---------------------------------------------------------------------------

interface CacheEntry {
  agent: Agent;
  history: HistoryData;           // same reference the agent holds
  jobManager: JobManager;         // main executor job manager
  worker: WorkerProcess;          // main executor worker
  language: string;
  persistedStepCount: number;
  lastActive: Date;
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

export class SessionManager implements ISessionManager {
  readonly sessionsDir: string;
  readonly db: InternalDatabase;

  private readonly rootWorkDir: string;
  private readonly dbPath: string;
  private readonly cache = new Map<string, CacheEntry>();
  private readonly running = new Map<string, AbortController>();

  constructor(workDir: string) {
    this.rootWorkDir = path.resolve(workDir);
    this.sessionsDir = path.join(this.rootWorkDir, "sessions");
    this.dbPath = path.join(this.rootWorkDir, "internal.db");
    fs.mkdirSync(this.sessionsDir, { recursive: true });
    this.db = new InternalDatabase(this.dbPath);
  }

  // ---------------------------------------------------------------------------
  // Session CRUD
  // ---------------------------------------------------------------------------

  async listSessions(): Promise<SessionSummary[]> {
    return this.db.listSessions().map((row) => ({
      session_id: row.session_id,
      name: row.name,
      last_accessed: row.last_accessed,
    }));
  }

  async createSession(name: string, config: SessionConfig): Promise<string> {
    const sessionId = randomUUID();
    const sessionDir = path.join(this.sessionsDir, sessionId);

    // 1. File system scaffold
    for (const sub of ["uploads", "outputs", "scripts", "internal"]) {
      fs.mkdirSync(path.join(sessionDir, sub), { recursive: true });
    }

    // 2. DB entry
    this.db.createSession(sessionId, name, sessionDir, config as Record<string, unknown>);
    this.db.saveSessionSpecialty(
      sessionId,
      (config.specialty_id as string | null) ?? null,
      (config.specialty_prompt as string | null) ?? null,
    );

    // 3. Init agent (may throw — roll back on failure)
    let entry: CacheEntry;
    try {
      entry = await this._initCacheEntry(sessionId, sessionDir, config);
    } catch (err) {
      fs.rmSync(sessionDir, { recursive: true, force: true });
      this.db.deleteSession(sessionId);
      throw err;
    }

    // 4. DB connection system step
    const connectionInfo = await this._detectConnection(entry);
    if (connectionInfo) {
      const sysStep = makeSystemStep(
        `Database connection established. Use variable '${connectionInfo.variable}' (${connectionInfo.type}) to query the database.`,
      );
      addStep(entry.history, sysStep);
    }

    // 5. Cache
    this.cache.set(sessionId, entry);

    // 6. Persist if we have the system step
    if (connectionInfo) {
      await this._saveSession(sessionId);
    }

    return sessionId;
  }

  async getSession(sessionId: string): Promise<SessionDetail | null> {
    const row = this.db.getSession(sessionId);
    if (!row) return null;
    const config = (this.db.getSessionConfig(sessionId) ?? {}) as SessionConfig;
    return {
      session_id: row.session_id,
      name: row.name,
      last_accessed: row.last_accessed,
      config,
    };
  }

  async updateSession(sessionId: string, name: string, config: SessionConfig): Promise<void> {
    const oldConfig = (this.db.getSessionConfig(sessionId) ?? {}) as SessionConfig;

    // Immutable keys guard
    for (const key of ["language", "python_bin", "r_home"] as const) {
      const oldVal = ((oldConfig[key] as string | undefined) ?? "").trim();
      const newVal = ((config[key] as string | undefined) ?? "").trim();
      if (oldVal && newVal && oldVal !== newVal) {
        throw new Error(
          `Cannot change '${key}' after session creation. ` +
          `Current: '${oldVal}', requested: '${newVal}'.`,
        );
      }
    }

    this.db.renameSession(sessionId, name);
    this.db.saveSessionConfig(sessionId, config as Record<string, unknown>);
    this.db.saveSessionSpecialty(
      sessionId,
      (config.specialty_id as string | null) ?? null,
      (config.specialty_prompt as string | null) ?? null,
    );

    const entry = this.cache.get(sessionId);
    if (!entry) return; // not loaded — new config will be used on next load

    const llmKeys = [
      "llm_provider", "llm_model", "llm_api_key", "llm_base_url",
      "llm_api_version", "temperature", "top_p", "reasoning_effort",
    ] as const;
    if (llmKeys.some((k) => oldConfig[k] !== config[k])) {
      // Agent doesn't expose engine swap; evict so next load uses the new config.
      this._evict(sessionId);
      return;
    }

    // Specialty prompt update
    const oldSpecialty = (oldConfig.specialty_prompt as string | undefined) ?? null;
    const newSpecialty = (config.specialty_prompt as string | undefined) ?? null;
    if (oldSpecialty !== newSpecialty) {
      // Agent doesn't expose memory.setSystemPrompt; evict for clean reload
      this._evict(sessionId);
      return;
    }

    // DB connection code update — inject into existing worker without restart
    const oldDbCode = ((oldConfig.db_connection_code as string | undefined) ?? "").trim();
    const newDbCode = ((config.db_connection_code as string | undefined) ?? "").trim();
    if (newDbCode && newDbCode !== oldDbCode) {
      try {
        await entry.worker.sendCommand("inject", { code: newDbCode });
        const connectionInfo = await this._detectConnection(entry);
        if (connectionInfo) {
          await this.addSystemMessage(
            sessionId,
            `Database connection established. Use variable '${connectionInfo.variable}' (${connectionInfo.type}) to query the database.`,
          );
        }
      } catch (err) {
        console.warn(`[SessionManager] DB code injection error for ${sessionId}:`, err);
      }
    }
  }

  async deleteSession(sessionId: string): Promise<void> {
    this._evict(sessionId);
    this.db.deleteSession(sessionId);
    const sessionDir = path.join(this.sessionsDir, sessionId);
    if (fs.existsSync(sessionDir)) fs.rmSync(sessionDir, { recursive: true });
  }

  async renameSession(sessionId: string, name: string): Promise<void> {
    this.db.renameSession(sessionId, name);
  }

  // ---------------------------------------------------------------------------
  // Chat & stop
  // ---------------------------------------------------------------------------

  getHistory(sessionId: string): Step[] {
    const entry = this.cache.get(sessionId);
    if (entry) {
      return entry.history.rounds.flatMap((r) => r.steps);
    }
    return this.db.getHistorySteps(sessionId);
  }

  async *chat(sessionId: string, message: string): AsyncGenerator<AgentEvent> {
    const entry = await this._getOrLoad(sessionId);
    if (!entry) throw new Error(`Session not found: ${sessionId}`);

    const ctl = new AbortController();
    this.running.set(sessionId, ctl);

    try {
      for await (const event of entry.agent.chat(message)) {
        if (ctl.signal.aborted) break;
        yield event;
      }
    } finally {
      this.running.delete(sessionId);
      await this._saveSession(sessionId);
    }
  }

  async stopSession(sessionId: string): Promise<boolean> {
    const ctl = this.running.get(sessionId);
    if (!ctl) return false;
    ctl.abort();
    return true;
  }

  async addSystemMessage(sessionId: string, message: string): Promise<void> {
    const entry = await this._getOrLoad(sessionId);
    if (!entry) throw new Error(`Session not found: ${sessionId}`);
    const sysStep = makeSystemStep(message);
    addStep(entry.history, sysStep);
    await this._saveSession(sessionId);
  }

  // ---------------------------------------------------------------------------
  // State inspection
  // ---------------------------------------------------------------------------

  async getVariables(sessionId: string): Promise<VariablesSnapshot | null> {
    const row = this.db.getSession(sessionId);
    if (!row) return null;

    const entry = await this._getOrLoad(sessionId);
    if (!entry) return { language: "python" };

    try {
      const data = await entry.jobManager.sendDirect("get_state", {});
      const vars = Array.isArray(data["variables"]) ? data["variables"] : [];
      return entry.language === "r"
        ? { r: vars, language: "r" }
        : { python: vars, language: "python" };
    } catch {
      return { language: entry.language };
    }
  }

  async getMemoryDebug(sessionId: string): Promise<Record<string, unknown> | null> {
    const row = this.db.getSession(sessionId);
    if (!row) return null;
    const entry = await this._getOrLoad(sessionId);
    if (!entry) return null;
    return entry.agent.getMemoryDebug() as unknown as Record<string, unknown>;
  }

  async getCompressionDebug(sessionId: string): Promise<Record<string, unknown> | null> {
    const row = this.db.getSession(sessionId);
    if (!row) return null;
    const entry = await this._getOrLoad(sessionId);
    if (!entry) return null;
    return entry.agent.getCompressionDebug() as unknown as Record<string, unknown>;
  }

  // ---------------------------------------------------------------------------
  // Private — cache helpers
  // ---------------------------------------------------------------------------

  private async _getOrLoad(sessionId: string): Promise<CacheEntry | null> {
    const cached = this.cache.get(sessionId);
    if (cached) {
      cached.lastActive = new Date();
      this.db.updateLastAccessed(sessionId);
      return cached;
    }
    return this._loadFromStorage(sessionId);
  }

  private async _loadFromStorage(sessionId: string): Promise<CacheEntry | null> {
    const row = this.db.getSession(sessionId);
    if (!row) return null;

    const config = (this.db.getSessionConfig(sessionId) ?? {}) as SessionConfig;

    let entry: CacheEntry;
    try {
      entry = await this._initCacheEntry(sessionId, row.work_dir, config);
    } catch (err) {
      // Re-throw with context so callers surface the real error instead of "Session not found".
      throw new Error(
        `Failed to initialize session ${sessionId}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    // Restore history from DB
    const storedSteps = this.db.getHistorySteps(sessionId);
    for (const step of storedSteps) {
      addStep(entry.history, step);
    }

    // Load executor state
    const statePath = path.join(row.work_dir, entry.language === "r" ? "state.RData" : "state.pkl");
    if (fs.existsSync(statePath)) {
      try {
        await entry.worker.sendCommand("load_state", { path: statePath });
      } catch (err) {
        console.warn(`[SessionManager] Failed to load state for ${sessionId}:`, err);
      }
    }

    // Restore agent memory (IndexedAgentMemory compression state)
    const sessionState = this.db.getSessionState(sessionId);
    if (sessionState?.agent_memory_data) {
      try {
        const memData = JSON.parse(sessionState.agent_memory_data) as Record<string, unknown>;
        // Memory is private in Agent; we can only reconstruct via history — no-op for now.
        void memData;
      } catch {
        // best-effort
      }
    }

    entry.persistedStepCount = storedSteps.length;
    this.cache.set(sessionId, entry);
    return entry;
  }

  private async _initCacheEntry(
    sessionId: string,
    sessionDir: string,
    config: SessionConfig,
  ): Promise<CacheEntry> {
    const language = ((config.language as string | undefined) ?? "python").toLowerCase();
    const pythonBin =
      (config.python_bin as string | undefined) ??
      process.env["MEDDS_PYTHON_BIN"] ??
      "python";

    const workerEnv: Record<string, string> = {};
    const rHome = (config.r_home as string | undefined) ?? process.env["MEDDS_R_HOME"];
    if (rHome) workerEnv["R_HOME"] = rHome;
    // Ensure python_worker/ is importable regardless of the server's cwd.
    const existingPythonPath = process.env["PYTHONPATH"] ?? "";
    workerEnv["PYTHONPATH"] = existingPythonPath ? `${_CORE_ROOT}:${existingPythonPath}` : _CORE_ROOT;

    const handlerClassPath =
      language === "r"
        ? "python_worker.handlers.RHandler"
        : "python_worker.handlers.PythonHandler";

    const workerConfig = { handlerClassPath, pythonBin, handlerKwargs: { work_dir: sessionDir }, env: workerEnv };
    const worker = await WorkerProcess.create(workerConfig);

    const jobManager = new JobManager(worker);

    // Inject DB connection code if provided
    const dbCode = ((config.db_connection_code as string | undefined) ?? "").trim();
    if (dbCode) {
      try {
        await worker.sendCommand("inject", { code: dbCode });
      } catch (err) {
        console.warn(`[SessionManager] DB code injection error for ${sessionId}:`, err);
      }
    }

    const executorTool =
      language === "r"
        ? new RExecutorTool(jobManager, worker.readyInfo)
        : new PythonExecutorTool(jobManager, worker.readyInfo);

    const engine = this._buildEngine(config);

    const agentMemory = new IndexedAgentMemory(engine, {
      startWindowSize: parseInt(process.env["MEMORY_START_WINDOW_SIZE"] ?? "5", 10),
      recentWindowSize: parseInt(process.env["MEMORY_RECENT_WINDOW_SIZE"] ?? "20", 10),
      compressThreshold: parseInt(process.env["MEMORY_COMPRESS_THRESHOLD"] ?? "2000", 10),
    });

    const history = makeHistory();

    const agent = new Agent(engine, history, {
      memory: agentMemory,
      specialtyPrompt: (config.specialty_prompt as string | undefined) ?? null,
      tools: [executorTool, new FileSystemTool(sessionDir)],
    });

    return {
      agent,
      history,
      jobManager,
      worker,
      language,
      persistedStepCount: 0,
      lastActive: new Date(),
    };
  }

  private _evict(sessionId: string): void {
    const entry = this.cache.get(sessionId);
    if (entry) {
      entry.worker.shutdown().catch(() => {});
      this.cache.delete(sessionId);
    }
  }

  // ---------------------------------------------------------------------------
  // Private — persistence
  // ---------------------------------------------------------------------------

  private async _saveSession(sessionId: string): Promise<void> {
    const entry = this.cache.get(sessionId);
    if (!entry) return;

    // 1. Persist executor state
    const statePath = path.join(
      this.sessionsDir,
      sessionId,
      entry.language === "r" ? "state.RData" : "state.pkl",
    );
    const relStatePath = `sessions/${sessionId}/${path.basename(statePath)}`;
    try {
      await entry.worker.sendCommand("save_state", { path: statePath });
      this.db.saveSessionState(
        sessionId,
        null,
        entry.language === "r" ? null : relStatePath,
        entry.language === "r" ? relStatePath : null,
      );
    } catch {
      // best-effort
    }

    // 2. Persist new history steps
    const allSteps = entry.history.rounds.flatMap((r) => r.steps);
    if (allSteps.length > entry.persistedStepCount) {
      let idx = 0;
      for (const round of entry.history.rounds) {
        for (const step of round.steps) {
          if (idx >= entry.persistedStepCount) {
            this.db.addHistoryStep(sessionId, round.roundNum, step);
          }
          idx++;
        }
      }
      entry.persistedStepCount = allSteps.length;
    }

    // 3. Touch timestamp
    this.db.updateLastAccessed(sessionId);
  }

  // ---------------------------------------------------------------------------
  // Private — LLM engine factory
  // ---------------------------------------------------------------------------

  private _buildEngine(config: SessionConfig): LLMEngine {
    const provider = ((config.llm_provider as string | undefined) ?? "openai").toLowerCase();
    const model = (config.llm_model as string | undefined) ?? "gpt-4o";
    const apiKey = config.llm_api_key as string | undefined;
    const baseURL = config.llm_base_url as string | undefined;
    const temperature = parseFloat(String(config.temperature ?? 1.0));
    const maxTokens = parseInt(process.env["LLM_MAX_NEW_TOKENS"] ?? "16384", 10);

    if (provider === "azure") {
      return createEngine({
        provider: "azure",
        model,
        apiKey: apiKey ?? "",
        endpoint: baseURL ?? "",
        apiVersion: (config.llm_api_version as string | undefined) ?? "2024-10-21",
        maxTokens,
        temperature,
      });
    }

    // openai, vllm, sglang, openrouter — all use the OpenAI-compatible API
    return createEngine({
      provider,
      model,
      apiKey: apiKey ?? "",
      ...(baseURL ? { baseURL } : {}),
      maxTokens,
      temperature,
    });
  }

  // ---------------------------------------------------------------------------
  // Private — detect DB connection from worker state
  // ---------------------------------------------------------------------------

  private async _detectConnection(
    entry: CacheEntry,
  ): Promise<{ variable: string; type: string } | null> {
    try {
      const data = await entry.worker.sendCommand("get_state", {});
      const variables = Array.isArray(data["variables"])
        ? (data["variables"] as Array<Record<string, string>>)
        : [];

      if (entry.language === "r") {
        const dbiClasses = new Set([
          "PqConnection", "MariaDBConnection", "MySQLConnection",
          "OraConnection", "SQLiteConnection", "OdbcConnection", "DBIConnection",
        ]);
        for (const v of variables) {
          const t = v["type"] ?? "";
          if ([...dbiClasses].some((c) => t.includes(c)) || t.includes("Connection")) {
            return { variable: v["name"] ?? "conn", type: t };
          }
        }
      } else {
        let result: { variable: string; type: string } | null = null;
        for (const v of variables) {
          const name = v["name"] ?? "";
          const t = v["type"] ?? "";
          if (name === "db_engine") return { variable: "db_engine", type: `SQLAlchemy (${t})` };
          if (name === "conn" && result === null) result = { variable: "conn", type: t };
        }
        return result;
      }
    } catch {
      // best-effort
    }
    return null;
  }
}
