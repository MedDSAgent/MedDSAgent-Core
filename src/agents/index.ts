import { loadPrompt, applyTemplate } from "../util/prompts.js";
import type { LLMEngine, ChatResult } from "../engines/index.js";
import type { HistoryData } from "../history/index.js";
import { makeUserStep, makeAgentStep, makeObservationStep, addStep } from "../history/index.js";
import type { AgentMemory, MemoryDebug, CompressionDebug } from "../memory/index.js";
import { FullHistoryAgentMemory } from "../memory/index.js";
import { Tool, AsyncTool, FinalResponseTool, JobWaitTool, JobCancelTool } from "../tools/index.js";
import type { JobManager } from "../jobs/index.js";
import { ToolBusyError } from "../jobs/index.js";

// ---------------------------------------------------------------------------
// Event types yielded by Agent.chat()
// ---------------------------------------------------------------------------

export interface EnrichedToolCall {
  name: string;
  arguments: Record<string, unknown>;
  toolTitle: string;
}

export type AgentEvent =
  | { type: "response"; data: string }
  | { type: "tool_calls"; data: EnrichedToolCall[] }
  | { type: "tool_running"; toolName: string; data: Record<string, unknown>; jobId: string | null }
  | { type: "tool_output"; data: string }
  | { type: "final_decision"; isFinal: boolean };

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

interface ParsedCall {
  name: string;
  arguments: Record<string, unknown>;
  tool: Tool | undefined;
}

type ToolOutputEntry = { tool_name: string; output: string } | null;

export interface AgentOptions {
  memory?: AgentMemory;
  specialtyPrompt?: string | null;
  tools?: Tool[];
  maxRetries?: number;
}

export class Agent {
  readonly agentId: string;
  private readonly tools: Tool[];
  private readonly memory: AgentMemory;
  private readonly maxRetries: number;

  constructor(
    private readonly engine: LLMEngine,
    private readonly history: HistoryData,
    options?: AgentOptions,
  ) {
    this.agentId = crypto.randomUUID();
    this.maxRetries = options?.maxRetries ?? 8;

    this.memory = options?.memory ?? new FullHistoryAgentMemory();

    const systemPromptTemplate = loadPrompt("System_prompt_template");
    const systemPrompt = applyTemplate(systemPromptTemplate, {});
    this.memory.setSystemPrompt(systemPrompt, options?.specialtyPrompt);

    // Auto-inject one JobWaitTool + JobCancelTool per unique JobManager
    const userTools = options?.tools ?? [];
    const seenManagers = new Set<JobManager>();
    const jobManagers: JobManager[] = [];
    for (const t of userTools) {
      if (t instanceof AsyncTool && !seenManagers.has(t.jobManager)) {
        seenManagers.add(t.jobManager);
        jobManagers.push(t.jobManager);
      }
    }

    const injected: Tool[] = [];
    for (const jm of jobManagers) {
      injected.push(new JobWaitTool(jm));
      injected.push(new JobCancelTool(jm));
    }

    this.tools = [...userTools, ...injected, new FinalResponseTool()];
  }

  reset(): void {
    this.memory.reset();
  }

  getTools(): readonly Tool[] {
    return this.tools;
  }

  getMemoryDebug(): MemoryDebug {
    return this.memory.getMemoryDebug(this.history);
  }

  getCompressionDebug(): CompressionDebug {
    return this.memory.getCompressionDebug(this.history);
  }

  // ---------------------------------------------------------------------------
  // Main entry point — single async streaming generator
  // ---------------------------------------------------------------------------

  async *chat(userInput: string): AsyncGenerator<AgentEvent> {
    const userStep = makeUserStep(userInput);
    addStep(this.history, userStep);
    await this.memory.onStepAddedAsync(userStep);

    let isFinalAnswer = false;

    while (true) {
      const messages = this.memory.getMessages(this.history);
      const toolSchemas = this.tools.map((t) => t.getToolCallSchema());

      let lastResponse = "";
      isFinalAnswer = false;

      for (let attempt = 0; attempt < this.maxRetries; attempt++) {
        try {
          const llmResult = await this.engine.chat(messages, toolSchemas);
          lastResponse = llmResult.response;

          // Yield interim text only when tool calls are also present
          if (llmResult.response && llmResult.toolCalls.length > 0) {
            yield { type: "response", data: llmResult.response };
          }

          // Yield tool_calls preview (exclude internal final_response)
          const visibleCalls = llmResult.toolCalls.filter((tc) => tc.name !== "final_response");
          if (visibleCalls.length > 0) {
            const enriched: EnrichedToolCall[] = visibleCalls.map((tc) => {
              const tool = this.tools.find((t) => t.name === tc.name);
              return {
                name: tc.name,
                arguments: tc.arguments,
                toolTitle: tool ? tool.getTitle(tc.arguments) : "",
              };
            });
            yield { type: "tool_calls", data: enriched };
          }

          // Execute tools, record steps — yield all events including final_decision
          for await (const event of this._processLLMResponse(llmResult)) {
            yield event;
            if (event.type === "final_decision") {
              isFinalAnswer = event.isFinal;
            }
          }

          break; // success — exit retry loop
        } catch (err: unknown) {
          const errMsg = err instanceof Error ? err.message : String(err);
          if (attempt < this.maxRetries - 1) {
            messages.push({ role: "assistant", content: lastResponse });
            messages.push({
              role: "user",
              content: `Error: ${errMsg}. Please regenerate the response fixing this error.`,
            });
          } else {
            const finalErrMsg = `Error: ${errMsg}. Unable to generate a valid response.`;
            yield { type: "response", data: `\n\n[System Error: ${finalErrMsg}]` };
            this._recordErrorAsFinal(finalErrMsg);
            return;
          }
        }
      }

      if (isFinalAnswer) return;
    }
  }

  // ---------------------------------------------------------------------------
  // Tool dispatch — one LLM turn
  // ---------------------------------------------------------------------------

  private async *_processLLMResponse(llmResult: ChatResult): AsyncGenerator<AgentEvent> {
    const { response: responseText, toolCalls } = llmResult;

    if (toolCalls.length === 0) {
      throw new Error(
        "No tool calls received. You must always respond by calling a tool. " +
          "Use final_response when you are ready to deliver your answer.",
      );
    }

    const parsed: ParsedCall[] = toolCalls.map((tc) => ({
      name: tc.name,
      arguments: tc.arguments,
      tool: this.tools.find((t) => t.name === tc.name),
    }));

    const finalCalls = parsed.filter((p) => p.name === "final_response");
    const realCalls = parsed.filter((p) => p.name !== "final_response");

    if (finalCalls.length > 0 && realCalls.length > 0) {
      throw new Error(
        "final_response must be called alone. " +
          "Do not mix it with other tool calls in the same response.",
      );
    }

    const isFinal = finalCalls.length > 0;

    const toolsList = parsed.map((p) => ({
      tool_name: p.name,
      tool_args: JSON.stringify(p.arguments),
      tool_title: p.tool ? p.tool.getTitle(p.arguments) : "",
    }));

    const agentStep = makeAgentStep(this.agentId, { response: responseText, tools: toolsList, isFinal });
    addStep(this.history, agentStep);
    await this.memory.onStepAddedAsync(agentStep);

    if (isFinal) {
      // Guard: cannot finalize while a background job is still running
      for (const t of this.tools) {
        if (t instanceof AsyncTool && t.jobManager.hasPending()) {
          const runningId = t.jobManager.getRunningJobId();
          throw new Error(
            `Cannot call final_response while job '${runningId}' is still running. ` +
              `Use job_wait(job_id='${runningId}', max_sec=<seconds>) to collect results, ` +
              `or job_cancel(job_id='${runningId}') to abort.`,
          );
        }
      }
      const firstFinal = finalCalls[0];
      const finalResponse = firstFinal
        ? String(firstFinal.arguments["response"] ?? "")
        : "";
      yield { type: "response", data: finalResponse };
      yield { type: "final_decision", isFinal: true };
      return;
    }

    // --- Phase 1: Dispatch all tools ---
    const toolOutputs: ToolOutputEntry[] = [];
    const pendingAsync: Array<{ idx: number; jobId: string; toolName: string }> = [];

    for (const p of realCalls) {
      const { name: tcName, arguments: args, tool } = p;

      if (tool === undefined) {
        const output = `Error: Tool '${tcName}' not found.`;
        toolOutputs.push({ tool_name: tcName, output });
        yield { type: "tool_output", data: output };
      } else if (tool instanceof AsyncTool) {
        try {
          const jobId = tool.submit(args);
          pendingAsync.push({ idx: toolOutputs.length, jobId, toolName: tcName });
          toolOutputs.push(null); // filled in Phase 2
          yield { type: "tool_running", toolName: tcName, data: args, jobId };
        } catch (err: unknown) {
          const output =
            err instanceof ToolBusyError
              ? `Error: ${err.message}`
              : `Error submitting '${tcName}': ${err instanceof Error ? err.message : String(err)}`;
          toolOutputs.push({ tool_name: tcName, output });
          yield { type: "tool_output", data: output };
        }
      } else {
        yield { type: "tool_running", toolName: tcName, data: args, jobId: null };
        let output: string;
        try {
          output = String(await tool.execute(args));
        } catch (err: unknown) {
          output = `Error executing tool '${tcName}': ${err instanceof Error ? err.message : String(err)}`;
        }
        toolOutputs.push({ tool_name: tcName, output });
        yield { type: "tool_output", data: output };
      }
    }

    // --- Phase 2: Auto-wait for async jobs ---
    const autoWaitSec = parseInt(process.env["MEDDS_AUTO_WAIT_TIMEOUT"] ?? "5", 10);

    for (const { idx, jobId, toolName: tcName } of pendingAsync) {
      const tool = this.tools.find((t) => t.name === tcName);
      const jm = tool instanceof AsyncTool ? tool.jobManager : null;

      let output: string;
      if (jm === null) {
        output = `Error: could not find job manager for '${tcName}'.`;
      } else {
        const job = await jm.waitAsync(jobId, autoWaitSec);
        if (job.status === "completed") {
          output = job.result ?? "(No output)";
        } else if (job.status === "failed") {
          output = `[Job failed]\n${job.error}`;
        } else if (job.status === "cancelled") {
          output = `Job '${jobId}' was cancelled.`;
        } else {
          const elapsed = (Date.now() - job.submittedAt.getTime()) / 1000;
          output =
            `Execution is still running (job_id: '${jobId}', elapsed: ${elapsed.toFixed(1)}s). ` +
            `Use job_wait(job_id='${jobId}', max_sec=<seconds>) to collect results ` +
            `when ready, or job_cancel(job_id='${jobId}') to abort.`;
        }
      }

      toolOutputs[idx] = { tool_name: tcName, output };
      yield { type: "tool_output", data: output };
    }

    // Record ObservationStep
    const completedOutputs = toolOutputs.filter(
      (o): o is { tool_name: string; output: string } => o !== null,
    );
    const obsStep = makeObservationStep(this.agentId, completedOutputs);
    addStep(this.history, obsStep);
    await this.memory.onStepAddedAsync(obsStep);

    yield { type: "final_decision", isFinal: false };
  }

  // ---------------------------------------------------------------------------
  // Error recovery
  // ---------------------------------------------------------------------------

  private _recordErrorAsFinal(errorMessage: string): void {
    const agentStep = makeAgentStep(this.agentId, {
      response: errorMessage,
      tools: [],
      isFinal: true,
    });
    addStep(this.history, agentStep);
    void this.memory.onStepAddedAsync(agentStep);
  }
}
