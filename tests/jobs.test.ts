import { describe, it, expect, vi } from "vitest";
import { JobManager, ToolBusyError } from "../src/jobs/index.js";
import type { WorkerProcess } from "../src/jobs/index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeWorker(
  result: Record<string, unknown> = { output: "42" },
  delay = 0,
): WorkerProcess {
  return {
    sendCommand: vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve(result), delay);
        }),
    ),
    restart: vi.fn().mockResolvedValue(undefined),
  };
}

// ---------------------------------------------------------------------------
// submit
// ---------------------------------------------------------------------------

describe("JobManager.submit", () => {
  it("returns a short job_id and sets status to running immediately", () => {
    const jm = new JobManager(makeWorker());
    const jobId = jm.submit("PythonExecutor", "execute", { code: "x=1" });
    expect(typeof jobId).toBe("string");
    expect(jobId.length).toBeGreaterThan(0);
    const job = jm.getJob(jobId);
    expect(job?.status).toBe("running");
    expect(jm.hasPending()).toBe(true);
    expect(jm.getRunningJobId()).toBe(jobId);
  });

  it("throws ToolBusyError when a job is already running", () => {
    const jm = new JobManager(makeWorker({}, 9999)); // never completes during test
    jm.submit("PythonExecutor", "execute", { code: "x=1" });
    expect(() => jm.submit("PythonExecutor", "execute", { code: "x=2" })).toThrow(ToolBusyError);
  });

  it("transitions to completed once the worker resolves", async () => {
    const jm = new JobManager(makeWorker({ output: "hello" }));
    const jobId = jm.submit("PythonExecutor", "execute", { code: "x=1" });
    const job = await jm.waitAsync(jobId, 2);
    expect(job.status).toBe("completed");
    expect(job.result).toBe("hello");
    expect(jm.hasPending()).toBe(false);
  });

  it("transitions to failed when the worker rejects", async () => {
    const worker: WorkerProcess = {
      sendCommand: vi.fn().mockRejectedValue(new Error("subprocess crashed")),
      restart: vi.fn().mockResolvedValue(undefined),
    };
    const jm = new JobManager(worker);
    const jobId = jm.submit("PythonExecutor", "execute", {});
    const job = await jm.waitAsync(jobId, 2);
    expect(job.status).toBe("failed");
    expect(job.error).toContain("subprocess crashed");
  });

  it("re-allows submission after the previous job completes", async () => {
    const jm = new JobManager(makeWorker({ output: "1" }));
    const id1 = jm.submit("PythonExecutor", "execute", {});
    await jm.waitAsync(id1, 2);
    // Should not throw
    const id2 = jm.submit("PythonExecutor", "execute", {});
    expect(id2).not.toBe(id1);
  });
});

// ---------------------------------------------------------------------------
// waitAsync
// ---------------------------------------------------------------------------

describe("JobManager.waitAsync", () => {
  it("returns timed_out job view when timeout expires before completion", async () => {
    const jm = new JobManager(makeWorker({}, 5000)); // slow worker
    const jobId = jm.submit("PythonExecutor", "execute", {});
    const job = await jm.waitAsync(jobId, 0.1); // 100ms timeout
    expect(job.status).toBe("timed_out");
    // Timed-out view is a snapshot — original job is still running
    expect(jm.getJob(jobId)?.status).toBe("running");
  });

  it("returns current status immediately when timeout <= 0", async () => {
    const jm = new JobManager(makeWorker({}, 5000));
    const jobId = jm.submit("PythonExecutor", "execute", {});
    const job = await jm.waitAsync(jobId, 0);
    expect(job.status).toBe("timed_out");
  });

  it("throws for an unknown job_id", async () => {
    const jm = new JobManager(makeWorker());
    await expect(jm.waitAsync("nonexistent", 1)).rejects.toThrow("not found");
  });

  it("returns immediately for a job that is already completed", async () => {
    const jm = new JobManager(makeWorker({ output: "done" }));
    const jobId = jm.submit("PythonExecutor", "execute", {});
    await jm.waitAsync(jobId, 2); // wait for completion
    // second wait — already completed
    const job = await jm.waitAsync(jobId, 0);
    expect(job.status).toBe("completed");
  });
});

// ---------------------------------------------------------------------------
// cancel
// ---------------------------------------------------------------------------

describe("JobManager.cancel", () => {
  it("sets job status to cancelled and calls worker.restart()", async () => {
    const worker = makeWorker({}, 9999);
    const jm = new JobManager(worker);
    const jobId = jm.submit("PythonExecutor", "execute", {});
    const job = jm.cancel(jobId);
    expect(job.status).toBe("cancelled");
    expect(jm.hasPending()).toBe(false);
    // restart is called asynchronously — wait a tick
    await new Promise((r) => setTimeout(r, 10));
    expect(worker.restart).toHaveBeenCalledOnce();
  });

  it("returns the job unchanged if it is not running", async () => {
    const jm = new JobManager(makeWorker({ output: "x" }));
    const jobId = jm.submit("PythonExecutor", "execute", {});
    await jm.waitAsync(jobId, 2);
    const job = jm.cancel(jobId);
    expect(job.status).toBe("completed"); // was already done, not changed to cancelled
  });

  it("cancelAll cancels the running job", () => {
    const jm = new JobManager(makeWorker({}, 9999));
    jm.submit("PythonExecutor", "execute", {});
    expect(jm.hasPending()).toBe(true);
    jm.cancelAll();
    expect(jm.hasPending()).toBe(false);
  });

  it("throws for an unknown job_id", () => {
    const jm = new JobManager(makeWorker());
    expect(() => jm.cancel("unknown")).toThrow("not found");
  });
});
