import { describe, it, expect } from "vitest";
import { spawnSync } from "child_process";
import * as os from "os";
import * as path from "path";
import * as fs from "fs";
import { WorkerProcess, WorkerStartupError, rWorkerSpec } from "../src/workers/WorkerProcess.js";

// ---------------------------------------------------------------------------
// Locate Rscript. Prefer the conda 'agent' env, fall back to PATH.
// These tests are skipped when no R is available; CI installs R so they run there.
// ---------------------------------------------------------------------------

function findRscript(): string | null {
  const candidates = [
    path.join(os.homedir(), "miniconda3", "envs", "agent", "bin", "Rscript"),
    path.join(os.homedir(), "anaconda3", "envs", "agent", "bin", "Rscript"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  const which = spawnSync("which", ["Rscript"], { encoding: "utf8", timeout: 5_000 });
  if (which.status === 0 && which.stdout.trim()) return which.stdout.trim();
  return null;
}

const RSCRIPT = findRscript();
const PROJECT_ROOT = path.resolve(import.meta.dirname, "..");
const ENTRY = path.join(PROJECT_ROOT, "r_worker", "entry.R");

async function makeWorker(workDir: string): Promise<WorkerProcess> {
  return WorkerProcess.create({
    spec: rWorkerSpec(ENTRY, RSCRIPT!),
    handlerKwargs: { work_dir: workDir },
    cwd: PROJECT_ROOT,
  });
}

function withTmp<T>(fn: (dir: string) => Promise<T>): () => Promise<T> {
  return async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "medds-r-test-"));
    for (const sub of ["uploads", "outputs", "scripts", "internal"]) {
      fs.mkdirSync(path.join(dir, sub), { recursive: true });
    }
    try {
      return await fn(dir);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  };
}

describe.skipIf(!RSCRIPT)("R worker E2E", () => {
  it(
    "spawns and reports r_version in readyInfo",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      expect(wp.isAlive).toBe(true);
      expect(String(wp.readyInfo["r_version"])).toMatch(/^R version/);
      await wp.shutdown();
      expect(wp.isAlive).toBe(false);
    }),
  );

  it(
    "reports available_libs as an array (never null), which RExecutorTool reads",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      // jsonlite turns a 0- or 1-element vector into null / a bare string unless
      // forced; RExecutorTool needs an array to build its tool description.
      expect(Array.isArray(wp.readyInfo["available_libs"])).toBe(true);
      await wp.shutdown();
    }),
  );

  it(
    "evaluates an expression and autoprints the result",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      const r = await wp.sendCommand("execute", { code: "1 + 1" });
      expect(String(r["output"]).trim()).toBe("[1] 2");
      await wp.shutdown();
    }),
  );

  it(
    "captures cat() output and persists variables across calls",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      const a = await wp.sendCommand("execute", { code: 'x <- 21; cat("hi\\n")' });
      expect(String(a["output"]).trim()).toBe("hi");
      const b = await wp.sendCommand("execute", { code: "x * 2" });
      expect(String(b["output"]).trim()).toBe("[1] 42");
      await wp.shutdown();
    }),
  );

  it(
    "does not autoprint invisible results",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      const r = await wp.sendCommand("execute", { code: "invisible(99)" });
      expect(String(r["output"]).trim()).toBe("(No output)");
      await wp.shutdown();
    }),
  );

  it(
    "reports runtime errors as [Error] without killing the worker",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      const r = await wp.sendCommand("execute", { code: 'stop("boom")' });
      expect(String(r["output"])).toContain("[Error]");
      expect(String(r["output"])).toContain("boom");
      // The worker must survive a user error.
      const ok = await wp.sendCommand("execute", { code: "1 + 1" });
      expect(String(ok["output"]).trim()).toBe("[1] 2");
      await wp.shutdown();
    }),
  );

  it(
    "reports parse errors as [Error]",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      const r = await wp.sendCommand("execute", { code: "this is not R (((" });
      expect(String(r["output"])).toContain("[Error]");
      await wp.shutdown();
    }),
  );

  it(
    "routes warnings to [stderr] while still returning the value",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      const r = await wp.sendCommand("execute", { code: 'as.numeric("abc")' });
      expect(String(r["output"])).toContain("[1] NA");
      expect(String(r["output"])).toContain("[stderr]");
      await wp.shutdown();
    }),
  );

  it(
    "get_state describes a data.frame as an HTML table",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      await wp.sendCommand("execute", {
        code: 'df <- data.frame(age = c(31, 45), sex = c("M", "F"), stringsAsFactors = FALSE)',
      });
      const state = (await wp.sendCommand("get_state", {})) as {
        variables: Array<Record<string, unknown>>;
      };
      const df = state.variables.find((v) => v["name"] === "df");
      expect(df).toBeDefined();
      expect(df!["type"]).toBe("data.frame");
      expect(df!["value"]).toBe("(2x2)");
      expect(String(df!["preview"])).toContain('class="dataframe df-table"');
      await wp.shutdown();
    }),
  );

  it(
    "get_state escapes HTML in data.frame cells",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      await wp.sendCommand("execute", {
        code: 'df <- data.frame(s = "<script>", stringsAsFactors = FALSE)',
      });
      const state = (await wp.sendCommand("get_state", {})) as {
        variables: Array<Record<string, unknown>>;
      };
      const df = state.variables.find((v) => v["name"] === "df");
      expect(String(df!["preview"])).toContain("&lt;script&gt;");
      expect(String(df!["preview"])).not.toContain("<script>");
      await wp.shutdown();
    }),
  );

  it(
    "get_state classifies factors by levels, not as atomic",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      await wp.sendCommand("execute", { code: 'f <- factor(c("a", "b", "a"))' });
      const state = (await wp.sendCommand("get_state", {})) as {
        variables: Array<Record<string, unknown>>;
      };
      const f = state.variables.find((v) => v["name"] === "f");
      expect(f!["type"]).toBe("factor");
      expect(f!["value"]).toBe("(3) [2 levels]");
      await wp.shutdown();
    }),
  );

  it(
    "get_state omits the injected *_DIR setup variables",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      await wp.sendCommand("execute", { code: "y <- 1" });
      const state = (await wp.sendCommand("get_state", {})) as {
        variables: Array<Record<string, unknown>>;
      };
      const names = state.variables.map((v) => v["name"]);
      expect(names).toContain("y");
      expect(names).not.toContain("WORK_DIR");
      expect(names).not.toContain("UPLOADS_DIR");
      await wp.shutdown();
    }),
  );

  it(
    "get_state returns an empty array for an empty environment",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      const state = (await wp.sendCommand("get_state", {})) as { variables: unknown };
      expect(Array.isArray(state.variables)).toBe(true);
      expect((state.variables as unknown[]).length).toBe(0);
      await wp.shutdown();
    }),
  );

  it(
    "inject runs code silently and returns an error string for bad code",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      const ok = await wp.sendCommand("inject", { code: "secret <- 123" });
      expect(ok["error"] ?? null).toBeNull();
      const seen = await wp.sendCommand("execute", { code: "secret" });
      expect(String(seen["output"]).trim()).toBe("[1] 123");

      const bad = await wp.sendCommand("inject", { code: "not valid R (((" });
      expect(typeof bad["error"]).toBe("string");
      await wp.shutdown();
    }),
  );

  it(
    "reset_state clears user variables but restores WORK_DIR",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      await wp.sendCommand("execute", { code: "gone <- 1" });
      await wp.sendCommand("reset_state", {});
      const r = await wp.sendCommand("execute", { code: 'exists("gone")' });
      expect(String(r["output"]).trim()).toBe("[1] FALSE");
      const wd = await wp.sendCommand("execute", { code: "WORK_DIR" });
      expect(String(wd["output"])).toContain(fs.realpathSync(dir));
      await wp.shutdown();
    }),
  );

  it(
    "save_state / load_state round-trips across a fresh worker",
    withTmp(async (dir) => {
      const statePath = path.join(dir, "state.RData");

      const w1 = await makeWorker(dir);
      await w1.sendCommand("execute", {
        code: 'df <- data.frame(a = 1:3); msg <- "persisted"',
      });
      await w1.sendCommand("save_state", { path: statePath });
      await w1.shutdown();
      expect(fs.existsSync(statePath)).toBe(true);

      const w2 = await makeWorker(dir);
      await w2.sendCommand("load_state", { path: statePath });
      const r = await w2.sendCommand("execute", { code: "cat(msg, nrow(df))" });
      expect(String(r["output"]).trim()).toBe("persisted 3");
      await w2.shutdown();
    }),
  );

  it(
    "load_state on a missing file is a no-op, not an error",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      await expect(
        wp.sendCommand("load_state", { path: path.join(dir, "nope.RData") }),
      ).resolves.toBeDefined();
      await wp.shutdown();
    }),
  );

  it(
    "restart clears state",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      await wp.sendCommand("execute", { code: "z <- 5" });
      await wp.restart();
      const r = await wp.sendCommand("execute", { code: 'exists("z")' });
      expect(String(r["output"]).trim()).toBe("[1] FALSE");
      await wp.shutdown();
    }),
  );

  it(
    "surfaces an unknown method as a worker error",
    withTmp(async (dir) => {
      const wp = await makeWorker(dir);
      await expect(wp.sendCommand("no_such_method", {})).rejects.toThrow(/unknown method/);
      await wp.shutdown();
    }),
  );

  it(
    "blocks system() when MEDDS_CODE_GATE is on",
    withTmp(async (dir) => {
      const wp = await WorkerProcess.create({
        spec: rWorkerSpec(ENTRY, RSCRIPT!),
        handlerKwargs: { work_dir: dir },
        env: { MEDDS_CODE_GATE: "true" },
        cwd: PROJECT_ROOT,
      });
      const r = await wp.sendCommand("execute", { code: 'system("echo pwned")' });
      expect(String(r["output"])).toContain("[Blocked]");
      await wp.shutdown();
    }),
  );

  it(
    "fails startup with WorkerStartupError when --work_dir is missing",
    async () => {
      await expect(
        WorkerProcess.create({
          spec: rWorkerSpec(ENTRY, RSCRIPT!),
          cwd: PROJECT_ROOT,
        }),
      ).rejects.toThrow(WorkerStartupError);
    },
  );
});
