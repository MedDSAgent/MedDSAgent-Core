import type { FastifyPluginAsync } from "fastify";
import type { ISessionManager } from "../types.js";

interface Opts {
  manager: ISessionManager;
}

export const healthRoutes: FastifyPluginAsync<Opts> = async (fastify, opts) => {
  const { manager } = opts;

  fastify.get("/", async () => {
    return { status: "ok", service: "MedDSAgent" };
  });

  fastify.get("/health", async () => {
    return {
      status: "ok",
      service: "MedDSAgent",
      runtime: "node",
      node_version: process.version,
      port: parseInt(process.env["PORT"] ?? "7842", 10),
    };
  });

  fastify.post("/workspace/init", async () => {
    const { mkdirSync } = await import("fs");
    mkdirSync(manager.sessionsDir, { recursive: true });
    return {
      work_dir: process.env["WORK_DIR"] ?? "./workspace",
      sessions_dir: manager.sessionsDir,
    };
  });
};
