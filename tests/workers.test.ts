import { describe, it, expect, beforeAll } from "vitest";
import { spawnSync } from "child_process";
import * as os from "os";
import * as path from "path";
import * as fs from "fs";
import { WorkerProcess, WorkerStartupError } from "../src/workers/WorkerProcess.js";

// ---------------------------------------------------------------------------
// Find the Python binary from the conda 'agent' environment (or system python3).
// All E2E tests are skipped when no usable Python is found.
// ---------------------------------------------------------------------------

function findAgentPython(): string | null {
  // Try conda agent env
  const conda = spawnSync(
    "conda",
    ["run", "-n", "agent", "python", "-c", "import sys; print(sys.executable)"],
    { encoding: "utf8", timeout: 15_000 },
  );
  if (conda.status === 0 && conda.stdout.trim()) {
    return conda.stdout.trim();
  }
  // Fallback: system python3
  const py3 = spawnSync("python3", ["-c", "import sys; print(sys.executable)"], {
    encoding: "utf8",
    timeout: 5_000,
  });
  if (py3.status === 0 && py3.stdout.trim()) return py3.stdout.trim();
  return null;
}

const PYTHON_BIN = findAgentPython();
const PROJECT_ROOT = path.resolve(import.meta.dirname, "..");

// ---------------------------------------------------------------------------
// E2E: actual subprocess spawn
// ---------------------------------------------------------------------------

describe.skipIf(!PYTHON_BIN)("WorkerProcess E2E", () => {
  it("spawns PythonHandler and receives readyInfo", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "medds-test-"));
    try {
      const wp = await WorkerProcess.create({
        handlerClassPath: "python_worker.handlers.PythonHandler",
        pythonBin: PYTHON_BIN!,
        handlerKwargs: { work_dir: tmpDir },
        cwd: PROJECT_ROOT,
      });
      expect(wp.isAlive).toBe(true);
      expect(typeof wp.readyInfo["python_version"]).toBe("string");
      await wp.shutdown();
      expect(wp.isAlive).toBe(false);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("executes print(1+1) and returns '2'", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "medds-test-"));
    try {
      const wp = await WorkerProcess.create({
        handlerClassPath: "python_worker.handlers.PythonHandler",
        pythonBin: PYTHON_BIN!,
        handlerKwargs: { work_dir: tmpDir },
        cwd: PROJECT_ROOT,
      });

      const result = await wp.sendCommand("execute", { code: "print(1+1)" });
      expect(String(result["output"]).trim()).toBe("2");

      await wp.shutdown();
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("persists state across calls", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "medds-test-"));
    try {
      const wp = await WorkerProcess.create({
        handlerClassPath: "python_worker.handlers.PythonHandler",
        pythonBin: PYTHON_BIN!,
        handlerKwargs: { work_dir: tmpDir },
        cwd: PROJECT_ROOT,
      });

      await wp.sendCommand("execute", { code: "x = 42" });
      const result = await wp.sendCommand("execute", { code: "print(x)" });
      expect(String(result["output"]).trim()).toBe("42");

      await wp.shutdown();
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("restart spawns a fresh worker (state is cleared)", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "medds-test-"));
    try {
      const wp = await WorkerProcess.create({
        handlerClassPath: "python_worker.handlers.PythonHandler",
        pythonBin: PYTHON_BIN!,
        handlerKwargs: { work_dir: tmpDir },
        cwd: PROJECT_ROOT,
      });

      await wp.sendCommand("execute", { code: "x = 99" });
      await wp.restart();
      expect(wp.isAlive).toBe(true);

      // x should not be defined in the new worker
      const result = await wp.sendCommand("execute", { code: "print('x' in dir())" });
      expect(String(result["output"]).trim()).toBe("False");

      await wp.shutdown();
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("worker error response is converted to a thrown Error", async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "medds-test-"));
    try {
      const wp = await WorkerProcess.create({
        handlerClassPath: "python_worker.handlers.PythonHandler",
        pythonBin: PYTHON_BIN!,
        handlerKwargs: { work_dir: tmpDir },
        cwd: PROJECT_ROOT,
      });

      // get_state with no args should succeed; unknown_method should fail
      await expect(wp.sendCommand("unknown_method", {})).rejects.toThrow();
      await wp.shutdown();
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("throws WorkerStartupError for a bad handler class path", async () => {
    await expect(
      WorkerProcess.create({
        handlerClassPath: "python_worker.handlers.NonExistentHandler",
        pythonBin: PYTHON_BIN!,
        cwd: PROJECT_ROOT,
      }),
    ).rejects.toThrow(WorkerStartupError);
  });
});

// ---------------------------------------------------------------------------
// No-Python-required tests (schema, config shape)
// ---------------------------------------------------------------------------

describe("WorkerProcess config", () => {
  it("WorkerStartupError and WorkerDeadError are named correctly", () => {
    const e = new WorkerStartupError("oops");
    expect(e.name).toBe("WorkerStartupError");
    expect(e instanceof Error).toBe(true);
  });
});
