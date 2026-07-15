// ---------------------------------------------------------------------------
// Job lifecycle management for async tools.
// ---------------------------------------------------------------------------

export type JobStatus = "running" | "completed" | "failed" | "timed_out" | "cancelled";

export interface Job {
  readonly jobId: string;
  readonly toolName: string;
  status: JobStatus;
  result?: string;
  error?: string;
  readonly submittedAt: Date;
  completedAt?: Date;
}

export class ToolBusyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ToolBusyError";
  }
}

// Minimal interface the JobManager needs from a worker subprocess.
// Phase 4 will provide a subprocess-backed implementation.
export interface WorkerProcess {
  sendCommand(method: string, params: Record<string, unknown>): Promise<Record<string, unknown>>;
  restart(): Promise<void>;
}

// ---------------------------------------------------------------------------
// JobManager
// ---------------------------------------------------------------------------

export class JobManager {
  private readonly jobs = new Map<string, Job>();
  private runningJobId: string | null = null;

  constructor(private readonly worker: WorkerProcess) {}

  // ------------------------------------------------------------------
  // Submit
  // ------------------------------------------------------------------

  submit(toolName: string, method: string, params: Record<string, unknown>): string {
    if (this.runningJobId !== null) {
      const running = this.jobs.get(this.runningJobId)!;
      throw new ToolBusyError(
        `${toolName} is busy — job '${this.runningJobId}' ` +
          `(submitted by ${running.toolName}) is still running. ` +
          `Use job_wait(job_id='${this.runningJobId}', max_sec=<seconds>) ` +
          `to collect results, or job_cancel(job_id='${this.runningJobId}') to abort.`,
      );
    }

    const jobId = crypto.randomUUID().slice(0, 8);
    const job: Job = { jobId, toolName, status: "running", submittedAt: new Date() };
    this.jobs.set(jobId, job);
    this.runningJobId = jobId;

    // Fire-and-forget: update the job object when the worker responds
    void this.worker
      .sendCommand(method, params)
      .then((data) => {
        if (job.status === "running") {
          job.result = String(data["output"] ?? "");
          job.status = "completed";
        }
      })
      .catch((err: unknown) => {
        if (job.status === "running") {
          job.error = err instanceof Error ? err.message : String(err);
          job.status = "failed";
        }
      })
      .finally(() => {
        job.completedAt = new Date();
        if (this.runningJobId === jobId) {
          this.runningJobId = null;
        }
      });

    return jobId;
  }

  // ------------------------------------------------------------------
  // Wait
  // ------------------------------------------------------------------

  async waitAsync(jobId: string, timeoutSec: number): Promise<Job> {
    const job = this.getOrRaise(jobId);
    if (job.status !== "running") return job;

    if (timeoutSec <= 0) {
      return job.status === "running" ? this.timedOutView(job) : job;
    }

    const pollMs = 200;
    const deadlineMs = Date.now() + timeoutSec * 1000;

    while (Date.now() < deadlineMs) {
      if (job.status !== "running") return job;
      await new Promise<void>((resolve) => setTimeout(resolve, pollMs));
    }

    return job.status === "running" ? this.timedOutView(job) : job;
  }

  // ------------------------------------------------------------------
  // Cancel
  // ------------------------------------------------------------------

  cancel(jobId: string): Job {
    const job = this.getOrRaise(jobId);
    if (job.status !== "running") return job;

    job.status = "cancelled";
    job.completedAt = new Date();
    if (this.runningJobId === jobId) {
      this.runningJobId = null;
    }

    void this.worker.restart().catch((err: unknown) => {
      console.warn("JobManager: worker restart failed after cancel:", err);
    });

    return job;
  }

  // ------------------------------------------------------------------
  // Queries
  // ------------------------------------------------------------------

  getJob(jobId: string): Job | undefined {
    return this.jobs.get(jobId);
  }

  hasPending(): boolean {
    return this.runningJobId !== null;
  }

  getRunningJobId(): string | null {
    return this.runningJobId;
  }

  /** Forward a non-job command directly to the worker (e.g. get_state, save_state). */
  async sendDirect(method: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.worker.sendCommand(method, params);
  }

  cancelAll(): void {
    if (this.runningJobId !== null) {
      try {
        this.cancel(this.runningJobId);
      } catch {
        // ignore — best-effort
      }
    }
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  private getOrRaise(jobId: string): Job {
    const job = this.jobs.get(jobId);
    if (job === undefined) {
      throw new Error(
        `Job '${jobId}' not found. It may have expired or the job_id is incorrect.`,
      );
    }
    return job;
  }

  private timedOutView(job: Job): Job {
    return {
      jobId: job.jobId,
      toolName: job.toolName,
      status: "timed_out",
      submittedAt: job.submittedAt,
    };
  }
}
