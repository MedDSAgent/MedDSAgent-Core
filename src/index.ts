// Package entry point.
export const VERSION = "0.2.0";

import { SessionManager } from "./session/index.js";
import { createApp, startServer as _startServer } from "./server/index.js";
export type { ISessionManager, ServerHandle } from "./server/index.js";
export type { SessionConfig } from "./server/types.js";
export { SessionManager, createApp };

// ---------------------------------------------------------------------------
// High-level convenience API — takes options instead of a pre-built manager.
// Used by the VS Code extension (in-process) and the CLI.
// ---------------------------------------------------------------------------

export interface MedDSAgentOpts {
  /** Root workspace directory (session data lives under <workDir>/sessions/). */
  workDir: string;
  /** HTTP port. Defaults to $PORT env var or 7842. */
  port?: number;
  /** HTTP host. Defaults to $HOST env var or '0.0.0.0'. */
  host?: string;
}

/**
 * Create a SessionManager without starting an HTTP server.
 * Useful when the caller drives sessions directly (tests, VS Code direct mode).
 */
export function createSessionManager(opts: Pick<MedDSAgentOpts, "workDir">): SessionManager {
  return new SessionManager(opts.workDir);
}

/**
 * Create a SessionManager + Fastify server and start listening.
 * Returns a handle with `address` and `close()`.
 */
export async function startServer(opts: MedDSAgentOpts): ReturnType<typeof _startServer> {
  const manager = new SessionManager(opts.workDir);
  return _startServer(
    manager,
    opts.port ?? parseInt(process.env["PORT"] ?? "7842", 10),
    opts.host ?? process.env["HOST"] ?? "0.0.0.0",
  );
}
