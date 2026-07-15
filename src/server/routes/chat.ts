import type { FastifyPluginAsync } from "fastify";
import { serializeStep } from "../../history/index.js";
import type { ISessionManager } from "../types.js";

interface Opts {
  manager: ISessionManager;
}

export const chatRoutes: FastifyPluginAsync<Opts> = async (fastify, opts) => {
  const { manager } = opts;

  // -------------------------------------------------------------------------
  // History
  // -------------------------------------------------------------------------

  fastify.get<{ Params: { session_id: string } }>(
    "/sessions/:session_id/history",
    async (request, reply) => {
      try {
        const steps = manager.getHistory(request.params.session_id);
        return { steps: steps.map(serializeStep) };
      } catch (err: unknown) {
        reply.code(500).send({
          detail: `Failed to fetch history: ${err instanceof Error ? err.message : String(err)}`,
        });
      }
    },
  );

  // -------------------------------------------------------------------------
  // Chat — SSE streaming
  // -------------------------------------------------------------------------

  fastify.post<{
    Params: { session_id: string };
    Body: { message: string; stream?: boolean };
  }>("/sessions/:session_id/chat", async (request, reply) => {
    const { session_id } = request.params;
    const { message, stream = false } = request.body;

    if (!stream) {
      // Non-streaming: consume the generator and wait for completion
      try {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        for await (const _event of manager.chat(session_id, message)) { /* drain */ }
        return { status: "success" };
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
      return;
    }

    // Streaming: SSE
    reply.hijack();
    const res = reply.raw;
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
    });

    try {
      for await (const event of manager.chat(session_id, message)) {
        if (res.destroyed) break;
        res.write(`data: ${JSON.stringify(event)}\n\n`);

        // After each tool output, push a variable state snapshot if available
        if (event.type === "tool_output") {
          try {
            const vars = await manager.getVariables(session_id);
            if (vars !== null && !res.destroyed) {
              res.write(
                `data: ${JSON.stringify({ type: "env_update", data: vars })}\n\n`,
              );
            }
          } catch {
            // snapshot is best-effort — ignore errors
          }
        }
      }
      if (!res.destroyed) {
        res.write(`data: ${JSON.stringify({ type: "done" })}\n\n`);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      if (!res.destroyed) {
        res.write(`data: ${JSON.stringify({ type: "error", data: msg })}\n\n`);
      }
    } finally {
      if (!res.destroyed) res.end();
    }
  });

  // -------------------------------------------------------------------------
  // Stop
  // -------------------------------------------------------------------------

  fastify.post<{ Params: { session_id: string } }>(
    "/sessions/:session_id/stop",
    async (request, reply) => {
      try {
        const stopped = await manager.stopSession(request.params.session_id);
        return stopped ? { status: "stopped" } : { status: "no_active_run" };
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // -------------------------------------------------------------------------
  // Variables
  // -------------------------------------------------------------------------

  fastify.get<{ Params: { session_id: string } }>(
    "/sessions/:session_id/variables",
    async (request, reply) => {
      try {
        const vars = await manager.getVariables(request.params.session_id);
        if (vars === null) {
          reply.code(404).send({ detail: "Session not found" });
          return;
        }
        return vars;
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // -------------------------------------------------------------------------
  // Memory inspection
  // -------------------------------------------------------------------------

  fastify.get<{ Params: { session_id: string } }>(
    "/sessions/:session_id/memory",
    async (request, reply) => {
      try {
        const debug = await manager.getMemoryDebug(request.params.session_id);
        if (debug === null) {
          reply.code(404).send({ detail: "Session not found" });
          return;
        }
        return debug;
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  fastify.get<{ Params: { session_id: string } }>(
    "/sessions/:session_id/memory/compression",
    async (request, reply) => {
      try {
        const debug = await manager.getCompressionDebug(request.params.session_id);
        if (debug === null) {
          reply.code(404).send({ detail: "Session not found" });
          return;
        }
        return debug;
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );
};
