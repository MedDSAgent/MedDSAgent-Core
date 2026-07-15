import type { ChatCompletionTool } from "openai/resources/chat/completions";
import type { JobManager } from "../jobs/index.js";

export type { ChatCompletionTool };

// ---------------------------------------------------------------------------
// Tool base class
// ---------------------------------------------------------------------------

export abstract class Tool {
  readonly toolType: "sync" | "async" = "sync";

  constructor(
    readonly name: string,
    readonly description: string,
  ) {}

  abstract execute(params: Record<string, unknown>): string | Promise<string>;
  abstract getToolCallSchema(): ChatCompletionTool;

  // Default: read LLM-provided 'title' key (code executor tools supply it).
  // Override for tools whose title is derived from structured params.
  getTitle(args: Record<string, unknown>): string {
    return typeof args["title"] === "string" ? args["title"] : "";
  }
}

// ---------------------------------------------------------------------------
// AsyncTool — submits work to a subprocess worker instead of executing inline
// ---------------------------------------------------------------------------

export abstract class AsyncTool extends Tool {
  override readonly toolType: "sync" | "async" = "async";

  constructor(
    name: string,
    description: string,
    readonly jobManager: JobManager,
  ) {
    super(name, description);
  }

  override execute(_params: Record<string, unknown>): string {
    throw new Error(
      `AsyncTool '${this.name}' does not support execute(). ` +
        "The agent loop calls submit() for async tools.",
    );
  }

  // Submit work to the worker subprocess and return a job_id immediately.
  abstract submit(params: Record<string, unknown>): string;
}

// ---------------------------------------------------------------------------
// FinalResponseTool — signals the end of a round
// ---------------------------------------------------------------------------

export class FinalResponseTool extends Tool {
  constructor() {
    super(
      "final_response",
      "Call this tool when you are ready to end the current round and deliver " +
        "your complete response to the user. This must be the only tool call in your " +
        "response — do not mix it with other tool calls.",
    );
  }

  override execute(params: Record<string, unknown>): string {
    return typeof params["response"] === "string" ? params["response"] : "";
  }

  override getTitle(_args: Record<string, unknown>): string {
    return "Final Response";
  }

  override getToolCallSchema(): ChatCompletionTool {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        parameters: {
          type: "object",
          properties: {
            response: {
              type: "string",
              description: "Your complete final response to the user.",
            },
          },
          required: ["response"],
        },
      },
    };
  }
}

// ---------------------------------------------------------------------------
// JobWaitTool — collect output from a running async job
// ---------------------------------------------------------------------------

export class JobWaitTool extends Tool {
  constructor(private readonly jobManager: JobManager) {
    super(
      "job_wait",
      "Wait for a running background job to complete and return its output. " +
        "Use this after receiving a job_id from PythonExecutor or RExecutor " +
        "when the job did not finish within the auto-wait window. " +
        "Pass max_sec=0 to instantly check the current status without waiting.",
    );
  }

  override async execute(params: Record<string, unknown>): Promise<string> {
    const jobId = String(params["job_id"] ?? "").trim();
    const maxSec = Number(params["max_sec"] ?? 60);

    if (!jobId) return "Error: job_id is required.";

    const job = await this.jobManager.waitAsync(jobId, maxSec);
    const elapsed = (Date.now() - job.submittedAt.getTime()) / 1000;

    switch (job.status) {
      case "completed":
        return job.result ?? "(No output)";
      case "failed":
        return `[Job failed]\n${job.error}`;
      case "cancelled":
        return `Job '${jobId}' was cancelled.`;
      case "timed_out":
        return (
          `Job '${jobId}' is still running (elapsed: ${elapsed.toFixed(1)}s, waited: ${maxSec}s). ` +
          `Call job_wait again with a longer max_sec, or job_cancel to abort.`
        );
      default:
        return `Job '${jobId}' status: ${job.status}.`;
    }
  }

  override getTitle(args: Record<string, unknown>): string {
    const jobId = typeof args["job_id"] === "string" ? args["job_id"] : "";
    return jobId ? `Wait for job ${jobId}` : "Wait for job";
  }

  override getToolCallSchema(): ChatCompletionTool {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        parameters: {
          type: "object",
          properties: {
            job_id: {
              type: "string",
              description: "The job_id returned by PythonExecutor or RExecutor.",
            },
            max_sec: {
              type: "number",
              description:
                "Maximum seconds to wait for the job to complete. " +
                "Use 0 for an instant status check. Default: 60.",
            },
          },
          required: ["job_id"],
        },
      },
    };
  }
}

// ---------------------------------------------------------------------------
// JobCancelTool — abort a running async job
// ---------------------------------------------------------------------------

export class JobCancelTool extends Tool {
  constructor(private readonly jobManager: JobManager) {
    super(
      "job_cancel",
      "Cancel a running background job. " +
        "The worker process is restarted — variables defined in the cancelled " +
        "execution will not be available. Use this when a job is taking too long " +
        "or you want to abandon the current execution.",
    );
  }

  override execute(params: Record<string, unknown>): string {
    const jobId = String(params["job_id"] ?? "").trim();
    if (!jobId) return "Error: job_id is required.";

    const job = this.jobManager.cancel(jobId);

    if (job.status === "cancelled") {
      return (
        `Job '${jobId}' has been cancelled. ` +
        `The executor has been restarted. ` +
        `Note: any variables defined in the cancelled execution are lost.`
      );
    }
    return (
      `Job '${jobId}' could not be cancelled (status: ${job.status}). ` +
      `It may have already finished.`
    );
  }

  override getTitle(args: Record<string, unknown>): string {
    const jobId = typeof args["job_id"] === "string" ? args["job_id"] : "";
    return jobId ? `Cancel job ${jobId}` : "Cancel job";
  }

  override getToolCallSchema(): ChatCompletionTool {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        parameters: {
          type: "object",
          properties: {
            job_id: {
              type: "string",
              description: "The job_id to cancel.",
            },
          },
          required: ["job_id"],
        },
      },
    };
  }
}
