import type { FastifyPluginAsync } from "fastify";
import type { ISessionManager, SessionConfig } from "../types.js";

interface Opts {
  manager: ISessionManager;
}

export const sessionsRoutes: FastifyPluginAsync<Opts> = async (fastify, opts) => {
  const { manager } = opts;

  // GET /sessions
  fastify.get("/sessions", async () => {
    return manager.listSessions();
  });

  // POST /sessions
  fastify.post<{ Body: { name: string; config: SessionConfig } }>(
    "/sessions",
    async (request, reply) => {
      try {
        const { name, config } = request.body;
        const sessionId = await manager.createSession(name, config);
        return { session_id: sessionId, name };
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // GET /sessions/:session_id
  fastify.get<{ Params: { session_id: string } }>(
    "/sessions/:session_id",
    async (request, reply) => {
      try {
        const session = await manager.getSession(request.params.session_id);
        if (!session) {
          reply.code(404).send({ detail: "Session not found" });
          return;
        }
        return session;
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // PUT /sessions/:session_id
  fastify.put<{ Params: { session_id: string }; Body: { name: string; config: SessionConfig } }>(
    "/sessions/:session_id",
    async (request, reply) => {
      try {
        const { name, config } = request.body;
        await manager.updateSession(request.params.session_id, name, config);
        return { status: "updated", session_id: request.params.session_id };
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // DELETE /sessions/:session_id
  fastify.delete<{ Params: { session_id: string } }>(
    "/sessions/:session_id",
    async (request, reply) => {
      try {
        await manager.deleteSession(request.params.session_id);
        return { status: "deleted", session_id: request.params.session_id };
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // PUT /sessions/:session_id/name
  fastify.put<{ Params: { session_id: string }; Body: { name: string } }>(
    "/sessions/:session_id/name",
    async (request, reply) => {
      try {
        await manager.renameSession(request.params.session_id, request.body.name);
        return { status: "renamed", name: request.body.name };
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // POST /test-db-connection
  // In TS the server cannot exec Python directly. Connection code is validated
  // when the session starts and the Python worker executes it.
  fastify.post<{ Body: { code: string } }>(
    "/test-db-connection",
    async (request, reply) => {
      const { code } = request.body;
      if (!code?.trim()) {
        reply.code(400).send({ detail: "No connection code provided" });
        return;
      }
      return {
        status: "success",
        message:
          "Connection code accepted. It will be executed when the session starts " +
          "and its result validated by the Python worker.",
        connection_type: "pending_validation",
      };
    },
  );
};
