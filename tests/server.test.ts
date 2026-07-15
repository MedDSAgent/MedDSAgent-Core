import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { createApp } from "../src/server/index.js";
import type { ISessionManager, SessionConfig, SessionSummary, SessionDetail, VariablesSnapshot } from "../src/server/types.js";
import type { InternalDatabase } from "../src/db/index.js";
import type { Step } from "../src/history/index.js";
import type { AgentEvent } from "../src/agents/index.js";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// ---------------------------------------------------------------------------
// Minimal mock database — only the methods filesRoutes uses
// ---------------------------------------------------------------------------

function makeMockDb(): InternalDatabase {
  return {
    getParsedDocument: () => null,
    getDocumentSections: () => [],
  } as unknown as InternalDatabase;
}

// ---------------------------------------------------------------------------
// Mock session manager
// ---------------------------------------------------------------------------

function makeMockManager(sessionsDir: string): ISessionManager {
  const sessions = new Map<string, { name: string; config: SessionConfig; accessed: string }>();

  return {
    sessionsDir,
    db: makeMockDb(),

    listSessions: async (): Promise<SessionSummary[]> =>
      [...sessions.entries()].map(([id, s]) => ({
        session_id: id,
        name: s.name,
        last_accessed: s.accessed,
      })),

    createSession: async (name: string, config: SessionConfig): Promise<string> => {
      const id = `sess-${Date.now()}`;
      sessions.set(id, { name, config, accessed: new Date().toISOString() });
      fs.mkdirSync(path.join(sessionsDir, id), { recursive: true });
      return id;
    },

    getSession: async (sessionId: string): Promise<SessionDetail | null> => {
      const s = sessions.get(sessionId);
      if (!s) return null;
      return { session_id: sessionId, name: s.name, last_accessed: s.accessed, config: s.config };
    },

    updateSession: async (sessionId: string, name: string, config: SessionConfig): Promise<void> => {
      const s = sessions.get(sessionId);
      if (s) sessions.set(sessionId, { name, config, accessed: s.accessed });
    },

    deleteSession: async (sessionId: string): Promise<void> => {
      sessions.delete(sessionId);
      const dir = path.join(sessionsDir, sessionId);
      if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true });
    },

    renameSession: async (sessionId: string, name: string): Promise<void> => {
      const s = sessions.get(sessionId);
      if (s) sessions.set(sessionId, { ...s, name });
    },

    getHistory: (_sessionId: string): Step[] => [],

    chat: async function* (_sessionId: string, _message: string): AsyncGenerator<AgentEvent> {
      yield { type: "response", data: "Hello from mock" };
    },

    stopSession: async (_sessionId: string): Promise<boolean> => false,

    addSystemMessage: async (_sessionId: string, _message: string): Promise<void> => {},

    getVariables: async (sessionId: string): Promise<VariablesSnapshot | null> => {
      if (!sessions.has(sessionId)) return null;
      return { python: [], r: [], language: "python" };
    },

    getMemoryDebug: async (sessionId: string): Promise<Record<string, unknown> | null> => {
      if (!sessions.has(sessionId)) return null;
      return { steps: 0 };
    },

    getCompressionDebug: async (sessionId: string): Promise<Record<string, unknown> | null> => {
      if (!sessions.has(sessionId)) return null;
      return { compressions: 0 };
    },
  };
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

describe("Fastify server", () => {
  let tmpDir: string;
  let app: ReturnType<typeof createApp>;
  let createdSessionId: string;

  beforeAll(async () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "medds-server-test-"));
    app = createApp(makeMockManager(tmpDir));
    await app.ready();
  });

  afterAll(async () => {
    await app.close();
    fs.rmSync(tmpDir, { recursive: true });
  });

  // ---- Health ----

  it("GET / returns ok", async () => {
    const res = await app.inject({ method: "GET", url: "/" });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ status: "ok" });
  });

  it("GET /health returns port", async () => {
    const res = await app.inject({ method: "GET", url: "/health" });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ status: "ok", service: "MedDSAgent" });
  });

  it("POST /workspace/init creates sessionsDir", async () => {
    const res = await app.inject({ method: "POST", url: "/workspace/init" });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toHaveProperty("sessions_dir");
  });

  // ---- Specialty prompts (returns empty list when no index.json) ----

  it("GET /specialty-prompts returns array", async () => {
    const res = await app.inject({ method: "GET", url: "/specialty-prompts" });
    expect(res.statusCode).toBe(200);
    expect(Array.isArray(res.json())).toBe(true);
  });

  it("GET /specialty-prompts/:id 404 for unknown id", async () => {
    const res = await app.inject({ method: "GET", url: "/specialty-prompts/nonexistent" });
    expect(res.statusCode).toBe(404);
  });

  // ---- Sessions CRUD ----

  it("GET /sessions returns empty array initially", async () => {
    const res = await app.inject({ method: "GET", url: "/sessions" });
    expect(res.statusCode).toBe(200);
    expect(Array.isArray(res.json())).toBe(true);
  });

  it("POST /sessions creates a session", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/sessions",
      payload: { name: "Test Session", config: { language: "python" } },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json<{ session_id: string; name: string }>();
    expect(body).toHaveProperty("session_id");
    expect(body.name).toBe("Test Session");
    createdSessionId = body.session_id;
  });

  it("GET /sessions/:id returns created session", async () => {
    const res = await app.inject({ method: "GET", url: `/sessions/${createdSessionId}` });
    expect(res.statusCode).toBe(200);
    const body = res.json<SessionDetail>();
    expect(body.session_id).toBe(createdSessionId);
    expect(body.name).toBe("Test Session");
  });

  it("GET /sessions/:id 404 for unknown session", async () => {
    const res = await app.inject({ method: "GET", url: "/sessions/ghost-session" });
    expect(res.statusCode).toBe(404);
  });

  it("PUT /sessions/:id updates session", async () => {
    const res = await app.inject({
      method: "PUT",
      url: `/sessions/${createdSessionId}`,
      payload: { name: "Renamed", config: { language: "r" } },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ status: "updated" });
  });

  it("PUT /sessions/:id/name renames session", async () => {
    const res = await app.inject({
      method: "PUT",
      url: `/sessions/${createdSessionId}/name`,
      payload: { name: "Final Name" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ status: "renamed", name: "Final Name" });
  });

  // ---- History & variables ----

  it("GET /sessions/:id/history returns steps array", async () => {
    const res = await app.inject({ method: "GET", url: `/sessions/${createdSessionId}/history` });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toHaveProperty("steps");
  });

  it("GET /sessions/:id/variables returns snapshot", async () => {
    const res = await app.inject({ method: "GET", url: `/sessions/${createdSessionId}/variables` });
    expect(res.statusCode).toBe(200);
    const body = res.json<VariablesSnapshot>();
    expect(body).toHaveProperty("python");
  });

  it("GET /sessions/:id/variables 404 for unknown session", async () => {
    const res = await app.inject({ method: "GET", url: "/sessions/ghost/variables" });
    expect(res.statusCode).toBe(404);
  });

  // ---- Memory debug ----

  it("GET /sessions/:id/memory returns debug info", async () => {
    const res = await app.inject({ method: "GET", url: `/sessions/${createdSessionId}/memory` });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toHaveProperty("steps");
  });

  it("GET /sessions/:id/memory/compression returns debug info", async () => {
    const res = await app.inject({
      method: "GET",
      url: `/sessions/${createdSessionId}/memory/compression`,
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toHaveProperty("compressions");
  });

  // ---- Stop ----

  it("POST /sessions/:id/stop returns no_active_run", async () => {
    const res = await app.inject({
      method: "POST",
      url: `/sessions/${createdSessionId}/stop`,
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ status: "no_active_run" });
  });

  // ---- Files ----

  it("GET /sessions/:id/files returns empty array for new session", async () => {
    const res = await app.inject({
      method: "GET",
      url: `/sessions/${createdSessionId}/files`,
    });
    expect(res.statusCode).toBe(200);
    expect(Array.isArray(res.json())).toBe(true);
  });

  it("GET /sessions/:id/files 404 for unknown session", async () => {
    const res = await app.inject({ method: "GET", url: "/sessions/ghost/files" });
    expect(res.statusCode).toBe(404);
  });

  it("GET /sessions/:id/files/content?path= 404 for missing file", async () => {
    const res = await app.inject({
      method: "GET",
      url: `/sessions/${createdSessionId}/files/content?path=no-such-file.txt`,
    });
    expect(res.statusCode).toBe(404);
  });

  it("DELETE /sessions/:id files 404 for missing file", async () => {
    const res = await app.inject({
      method: "DELETE",
      url: `/sessions/${createdSessionId}/files/no-such-file.txt`,
    });
    expect(res.statusCode).toBe(404);
  });

  it("GET /sessions/:id/files/:name/index-status returns not_indexed", async () => {
    const res = await app.inject({
      method: "GET",
      url: `/sessions/${createdSessionId}/files/foo.pdf/index-status`,
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ status: "not_indexed" });
  });

  it("POST /test-db-connection returns pending_validation", async () => {
    const res = await app.inject({
      method: "POST",
      url: "/test-db-connection",
      payload: { code: "import sqlalchemy" },
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ status: "success", connection_type: "pending_validation" });
  });

  // ---- Session delete (last so other tests still have the session) ----

  it("DELETE /sessions/:id deletes the session", async () => {
    const res = await app.inject({
      method: "DELETE",
      url: `/sessions/${createdSessionId}`,
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({ status: "deleted" });
  });
});
