import { spawn } from "child_process";
import type { ChildProcess } from "child_process";
import * as readline from "readline";

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

export class WorkerStartupError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WorkerStartupError";
  }
}

export class WorkerDeadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WorkerDeadError";
  }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/**
 * How to launch a worker. The IPC contract (docs/worker-protocol.md) is
 * language-agnostic, so this describes the command only — nothing here assumes
 * Python. See pythonWorkerSpec() / rWorkerSpec() for the concrete launchers.
 */
export interface WorkerSpec {
  /** Executable to spawn, e.g. "python" or "Rscript". */
  command: string;
  /** Fixed args preceding the --key value handler kwargs. */
  args: string[];
}

/** Launch spec for a handler in the Python worker (python_worker/). */
export function pythonWorkerSpec(handlerClassPath: string, pythonBin = "python"): WorkerSpec {
  return { command: pythonBin, args: ["-m", "python_worker.entry", handlerClassPath] };
}

/** Launch spec for the R worker (r_worker/). `entryPath` is r_worker/entry.R. */
export function rWorkerSpec(entryPath: string, rscriptBin = "Rscript"): WorkerSpec {
  return { command: rscriptBin, args: [entryPath] };
}

export interface WorkerProcessConfig {
  /** How to launch the worker. */
  spec: WorkerSpec;
  /** Key-value pairs forwarded as --key value CLI args to the handler. */
  handlerKwargs?: Record<string, string>;
  /** Extra env vars merged on top of process.env. */
  env?: Record<string, string>;
  /** Working directory for the subprocess. Defaults to process.cwd(). */
  cwd?: string;
}

// ---------------------------------------------------------------------------
// WorkerProcess
// Implements the WorkerProcess interface from src/jobs/index.ts.
// ---------------------------------------------------------------------------

export class WorkerProcess {
  private child: ChildProcess | null = null;
  private rl: readline.Interface | null = null;
  private dead = true;

  // Pending readline resolvers — resolved/rejected as lines arrive or process dies
  private pendingReads: Array<{
    resolve: (line: string) => void;
    reject: (err: Error) => void;
  }> = [];
  private lineBuffer: string[] = [];

  // Serializes all sendCommand calls (one JSON exchange at a time)
  private cmdQueue: Promise<unknown> = Promise.resolve();

  /** Startup handshake payload returned by the handler. */
  readonly readyInfo: Record<string, unknown> = {};

  private constructor(private readonly config: WorkerProcessConfig) {}

  /** Spawn the subprocess and perform the startup handshake. */
  static async create(config: WorkerProcessConfig): Promise<WorkerProcess> {
    const wp = new WorkerProcess(config);
    await wp._start();
    return wp;
  }

  // ------------------------------------------------------------------
  // WorkerProcess interface (implements src/jobs/index.ts#WorkerProcess)
  // ------------------------------------------------------------------

  async sendCommand(
    method: string,
    params: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this._enqueue(async () => {
      if (this.dead || this.child === null) {
        throw new WorkerDeadError("Worker subprocess is not running.");
      }

      const payload = JSON.stringify({ method, params }) + "\n";

      await new Promise<void>((resolve, reject) => {
        this.child!.stdin!.write(payload, (err) => {
          if (err) reject(new WorkerDeadError(`Failed to write to worker stdin: ${err.message}`));
          else resolve();
        });
      });

      const raw = await this._readLine();

      let response: Record<string, unknown>;
      try {
        response = JSON.parse(raw) as Record<string, unknown>;
      } catch {
        throw new WorkerDeadError(`Worker returned invalid JSON: ${JSON.stringify(raw)}`);
      }

      if (response["status"] !== "ok") {
        throw new Error(
          `Worker error for '${method}': ${String(response["error"] ?? "unknown")}`,
        );
      }

      return (response["data"] ?? {}) as Record<string, unknown>;
    });
  }

  async restart(): Promise<void> {
    await this._terminate();
    this.cmdQueue = Promise.resolve(); // reset serialization queue
    await this._start();
  }

  // ------------------------------------------------------------------
  // Additional lifecycle methods
  // ------------------------------------------------------------------

  get isAlive(): boolean {
    return !this.dead && this.child !== null;
  }

  async shutdown(): Promise<void> {
    if (this.child === null || this.dead) return;
    try {
      await this.sendCommand("shutdown", {});
    } catch {
      // ignore — process might be dying already
    }
    await this._terminate();
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  private _buildArgs(): [string, string[]] {
    const { command, args: specArgs } = this.config.spec;
    const args = [...specArgs];
    for (const [k, v] of Object.entries(this.config.handlerKwargs ?? {})) {
      args.push(`--${k}`, v);
    }
    return [command, args];
  }

  private async _start(): Promise<void> {
    const [command, args] = this._buildArgs();
    const env = this.config.env ? { ...process.env, ...this.config.env } : undefined;

    this.child = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env,
      cwd: this.config.cwd ?? process.cwd(),
    });
    this.dead = false; // mark alive immediately so _readLine() doesn't short-circuit

    // Route stdout lines to pending reads or buffer
    this.rl = readline.createInterface({ input: this.child.stdout! });
    this.rl.on("line", (line) => {
      const pending = this.pendingReads.shift();
      if (pending !== undefined) {
        pending.resolve(line);
      } else {
        this.lineBuffer.push(line);
      }
    });

    // Reject all pending reads when the process exits
    this.child.on("close", (code) => {
      this.dead = true;
      const err = new WorkerDeadError(
        `Worker process exited unexpectedly (code ${code ?? "unknown"}).`,
      );
      const waiters = this.pendingReads.splice(0);
      for (const w of waiters) w.reject(err);
    });

    // Prevent spawn errors (e.g. bad executable path) from becoming unhandled exceptions
    // that crash the server. The close event will fire and reject pending reads normally.
    this.child.on("error", (err) => {
      this.dead = true;
      const waiters = this.pendingReads.splice(0);
      for (const w of waiters) w.reject(err);
    });

    // Read startup handshake (one JSON line)
    let raw: string;
    try {
      raw = await this._readLine();
    } catch (err) {
      const stderr = await this._drainStderr();
      throw new WorkerStartupError(
        `Failed to read startup handshake: ${err instanceof Error ? err.message : String(err)}` +
          (stderr ? `\nstderr:\n${stderr}` : ""),
      );
    }

    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      throw new WorkerStartupError(`Invalid startup JSON from worker: ${JSON.stringify(raw)}`);
    }

    if (msg["status"] !== "ok") {
      throw new WorkerStartupError(
        `Worker reported startup error: ${String(msg["error"] ?? "unknown")}`,
      );
    }

    const data = msg["data"];
    if (data !== null && typeof data === "object") {
      Object.assign(this.readyInfo, data);
    }
  }

  private async _terminate(): Promise<void> {
    this.dead = true;
    const err = new WorkerDeadError("Worker is being terminated.");
    const waiters = this.pendingReads.splice(0);
    for (const w of waiters) w.reject(err);
    this.lineBuffer = [];

    if (this.rl !== null) {
      this.rl.close();
      this.rl = null;
    }

    if (this.child !== null) {
      const child = this.child;
      this.child = null;
      await new Promise<void>((resolve) => {
        try {
          child.kill("SIGTERM");
        } catch {
          // ignore — already dead
        }
        const timeout = setTimeout(() => {
          try { child.kill("SIGKILL"); } catch { /* ignore */ }
          resolve();
        }, 3000);
        child.once("close", () => {
          clearTimeout(timeout);
          resolve();
        });
      });
    }
  }

  private _readLine(): Promise<string> {
    const buffered = this.lineBuffer.shift();
    if (buffered !== undefined) return Promise.resolve(buffered);
    if (this.dead) return Promise.reject(new WorkerDeadError("Worker is dead."));
    return new Promise<string>((resolve, reject) => {
      this.pendingReads.push({ resolve, reject });
    });
  }

  private _enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.cmdQueue.then(() => fn());
    this.cmdQueue = result.catch(() => {});
    return result;
  }

  private _drainStderr(): Promise<string> {
    return new Promise((resolve) => {
      if (this.child?.stderr == null) { resolve(""); return; }
      let buf = "";
      this.child.stderr.on("data", (d: Buffer) => { buf += d.toString(); });
      this.child.stderr.once("end", () => resolve(buf));
      setTimeout(() => resolve(buf), 500);
    });
  }
}
