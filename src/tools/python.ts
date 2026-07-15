import type { ChatCompletionTool } from "openai/resources/chat/completions";
import type { JobManager } from "../jobs/index.js";
import { AsyncTool } from "./index.js";

// ---------------------------------------------------------------------------
// PythonExecutorTool
// ---------------------------------------------------------------------------

export class PythonExecutorTool extends AsyncTool {
  constructor(jobManager: JobManager, readyInfo: Record<string, unknown> = {}) {
    const pythonVersion =
      typeof readyInfo["python_version"] === "string"
        ? readyInfo["python_version"]
        : "Python";

    const availableLibs = Array.isArray(readyInfo["available_libs"])
      ? (readyInfo["available_libs"] as string[]).join(", ")
      : "standard library";

    const description =
      `Executes ${pythonVersion} code with full permissions. ` +
      `State persists across calls (variables, imports, DataFrames). ` +
      `Use \`print()\` to output results. ` +
      `Pre-imported: ${availableLibs}. ` +
      `Directories: UPLOADS_DIR for user files, OUTPUTS_DIR for agent outputs, ` +
      `SCRIPTS_DIR for scripts shared with user, INTERNAL_DIR for internal use. ` +
      `For database connections, use the pre-configured \`db_engine\` or \`conn\` object if available in state.`;

    super("PythonExecutor", description, jobManager);
  }

  override submit(params: Record<string, unknown>): string {
    const code = params["code"];
    if (typeof code !== "string") {
      throw new Error("PythonExecutorTool: 'code' parameter must be a string.");
    }
    return this.jobManager.submit(this.name, "execute", { code });
  }

  override getTitle(args: Record<string, unknown>): string {
    return typeof args["title"] === "string" ? args["title"] : "";
  }

  /** Fetch current variable state from the Python worker (for the env panel in the UI). */
  async getState(): Promise<unknown[]> {
    const data = await this.jobManager.sendDirect("get_state", {});
    return Array.isArray(data["variables"]) ? (data["variables"] as unknown[]) : [];
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
            title: {
              type: "string",
              description:
                "A one-line summary of what this code does, shown in the UI when the code block is collapsed. " +
                "Be specific and concise (e.g., 'Load data from uploads/data.csv and inspect shape', " +
                "'Fit logistic regression for mortality with predictors age, sex, and BMI', " +
                "'Plot Kaplan-Meier survival curve by treatment group').",
            },
            code: {
              type: "string",
              description: "Python code to execute.",
            },
          },
          required: ["title", "code"],
        },
      },
    };
  }
}

// ---------------------------------------------------------------------------
// RExecutorTool
// ---------------------------------------------------------------------------

export class RExecutorTool extends AsyncTool {
  constructor(jobManager: JobManager, readyInfo: Record<string, unknown> = {}) {
    const rVersion =
      typeof readyInfo["r_version"] === "string" ? readyInfo["r_version"] : "R";

    // Key must match what r_worker/entry.R sends in its ready_info — and what
    // PythonExecutorTool reads. This previously looked for "available_packages",
    // which no worker has ever sent, so it always fell back to "base packages".
    const availablePackages =
      Array.isArray(readyInfo["available_libs"]) && readyInfo["available_libs"].length > 0
        ? (readyInfo["available_libs"] as string[]).join(", ")
        : "base packages";

    const description =
      `Executes ${rVersion} code with full permissions. ` +
      `State persists across calls (variables, data frames, models). ` +
      `Use \`print()\` or \`cat()\` to output results. ` +
      `Pre-loaded packages: ${availablePackages}. ` +
      `Directories: UPLOADS_DIR for user files, OUTPUTS_DIR for agent outputs, ` +
      `SCRIPTS_DIR for scripts shared with user, INTERNAL_DIR for internal use. ` +
      `For database connections, use the pre-configured connection object if available.`;

    super("RExecutor", description, jobManager);
  }

  override submit(params: Record<string, unknown>): string {
    const code = params["code"];
    if (typeof code !== "string") {
      throw new Error("RExecutorTool: 'code' parameter must be a string.");
    }
    return this.jobManager.submit(this.name, "execute", { code });
  }

  override getTitle(args: Record<string, unknown>): string {
    return typeof args["title"] === "string" ? args["title"] : "";
  }

  async getState(): Promise<unknown[]> {
    const data = await this.jobManager.sendDirect("get_state", {});
    return Array.isArray(data["variables"]) ? (data["variables"] as unknown[]) : [];
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
            title: {
              type: "string",
              description:
                "A one-line summary of what this code does, shown in the UI when the code block is collapsed.",
            },
            code: {
              type: "string",
              description: "R code to execute.",
            },
          },
          required: ["title", "code"],
        },
      },
    };
  }
}
