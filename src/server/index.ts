import Fastify from "fastify";
import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import type { ISessionManager } from "./types.js";
import { healthRoutes } from "./routes/health.js";
import { specialtyRoutes } from "./routes/specialty.js";
import { sessionsRoutes } from "./routes/sessions.js";
import { chatRoutes } from "./routes/chat.js";
import { filesRoutes } from "./routes/files.js";

export { type ISessionManager } from "./types.js";

export function createApp(manager: ISessionManager) {
  const app = Fastify({ logger: false });

  void app.register(cors, { origin: "*" });
  void app.register(multipart, { limits: { fileSize: 500 * 1024 * 1024 } }); // 500 MB

  void app.register(healthRoutes, { manager });
  void app.register(specialtyRoutes);
  void app.register(sessionsRoutes, { manager });
  void app.register(chatRoutes, { manager });
  void app.register(filesRoutes, { manager });

  return app;
}

export interface ServerHandle {
  address: string;
  close(): Promise<void>;
}

export async function startServer(
  manager: ISessionManager,
  port = parseInt(process.env["PORT"] ?? "7842", 10),
  host = process.env["HOST"] ?? "0.0.0.0",
): Promise<ServerHandle> {
  const app = createApp(manager);
  const address = await app.listen({ port, host });
  console.log(`MedDSAgent server listening at ${address}`);
  return { address, close: () => app.close() };
}
