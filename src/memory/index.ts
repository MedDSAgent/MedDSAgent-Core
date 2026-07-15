import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import type { HistoryData, Step, UserStep, AgentStep, ObservationStep, SystemStep } from "../history/index.js";
import type { LLMEngine } from "../engines/index.js";
import { loadPrompt, applyTemplate } from "../util/prompts.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DebugEntry {
  index: number;
  role: string;
  content: string;
  content_length: number;
  step_id: string | null;
  round_num: number | null;
  step_type: string;
  is_compressed: boolean;
  original_content_length?: number;
}

export interface MemoryDebug {
  memory_type: string;
  config: Record<string, unknown>;
  summary: Record<string, unknown>;
  messages: DebugEntry[];
}

export interface CompressionDebug {
  memory_type: string;
  compression_supported: boolean;
  config?: Record<string, unknown>;
  total_compressed?: number;
  pending_compressions?: number;
  entries: Array<{
    step_id: string;
    step_type: string;
    round_num: number;
    original_content: string;
    compressed_content: string;
    original_length: number;
    compressed_length: number;
    compression_ratio: number | null;
  }>;
}

// ---------------------------------------------------------------------------
// AgentMemory — abstract base
// ---------------------------------------------------------------------------

export abstract class AgentMemory {
  protected systemPrompt = "";

  // Prompt templates — loaded lazily the first time they're needed
  private _userStepTemplate?: string;
  private _toolCallTemplate?: string;
  private _agentStepToolcallTemplate?: string;

  protected get userStepTemplate(): string {
    return (this._userStepTemplate ??= loadPrompt("UserStep_prompt_template"));
  }
  protected get toolCallTemplate(): string {
    return (this._toolCallTemplate ??= loadPrompt("ObservationStep_toolcall_prompt_template"));
  }
  protected get agentStepToolcallTemplate(): string {
    return (this._agentStepToolcallTemplate ??= loadPrompt("AgentStep_toolcall_prompt_template"));
  }

  setSystemPrompt(systemPrompt: string, specialtyPrompt?: string | null): void {
    this.systemPrompt = systemPrompt;
    if (specialtyPrompt) this.systemPrompt += `\n\n${specialtyPrompt}`;
  }

  abstract getMessages(history: HistoryData): ChatCompletionMessageParam[];
  abstract onStepAdded(step: Step): void;
  abstract onStepAddedAsync(step: Step): Promise<void>;
  abstract serialize(): Record<string, unknown>;
  abstract deserialize(data: Record<string, unknown>): void;
  abstract reset(): void;

  // ------------------------------------------------------------------
  // Formatting helpers
  // ------------------------------------------------------------------

  protected formatUserStep(step: UserStep, systemSteps: SystemStep[]): ChatCompletionMessageParam {
    const systemMessage =
      systemSteps.length > 0
        ? systemSteps.map((s) => s.systemMessage).join("\n")
        : "No system messages.";
    return {
      role: "user",
      content: applyTemplate(this.userStepTemplate, {
        requested_time: step.startTime.replace("T", " ").slice(0, 19),
        system_message: systemMessage,
        user_input: step.userInput,
      }),
    };
  }

  protected formatObservationStep(step: ObservationStep): ChatCompletionMessageParam {
    const header = "## System Message\nThe tool-calls from the previous step were executed.\n\n";
    const blocks = step.toolOutputs.map((to) =>
      applyTemplate(this.toolCallTemplate, {
        tool_name: to.tool_name,
        output: to.output,
      }),
    );
    return { role: "user", content: header + blocks.join("\n\n") };
  }

  protected formatAgentStep(step: AgentStep): ChatCompletionMessageParam {
    let content: string;
    if (step.tools.length > 0) {
      const blocks = step.tools.map((t) =>
        applyTemplate(this.agentStepToolcallTemplate, {
          tool_name: t.tool_name,
          tool_args: t.tool_args,
        }),
      );
      content = blocks.join("\n\n");
      if (step.response) content = step.response + "\n\n" + content;
    } else {
      content = step.response;
    }
    return { role: "assistant", content };
  }

  // ------------------------------------------------------------------
  // Debug helpers
  // ------------------------------------------------------------------

  protected buildDebugEntry(
    index: number,
    msg: ChatCompletionMessageParam,
    opts?: {
      step?: Step;
      roundNum?: number;
      stepType?: string;
      isCompressed?: boolean;
      originalContentLength?: number;
    },
  ): DebugEntry {
    const content = (msg.content as string) ?? "";
    const entry: DebugEntry = {
      index,
      role: msg.role,
      content,
      content_length: content.length,
      step_id: opts?.step?.stepId ?? null,
      round_num: opts?.roundNum ?? null,
      step_type: opts?.stepType ?? "system_prompt",
      is_compressed: opts?.isCompressed ?? false,
    };
    if (opts?.originalContentLength != null) {
      entry.original_content_length = opts.originalContentLength;
    }
    return entry;
  }

  getMemoryDebug(history: HistoryData): MemoryDebug {
    const messages = this.getMessages(history);
    const debugMessages = messages.map((msg, i) => ({
      index: i,
      role: msg.role,
      content: (msg.content as string) ?? "",
      content_length: ((msg.content as string) ?? "").length,
      step_id: null,
      round_num: null,
      step_type: i === 0 ? "system_prompt" : "unknown",
      is_compressed: false,
    }));
    return {
      memory_type: this.constructor.name,
      config: {},
      summary: {
        total_rounds: history.rounds.length,
        total_messages: debugMessages.length,
        total_content_length: debugMessages.reduce((s, m) => s + m.content_length, 0),
      },
      messages: debugMessages,
    };
  }

  getCompressionDebug(_history: HistoryData): CompressionDebug {
    return { memory_type: this.constructor.name, compression_supported: false, entries: [] };
  }
}

// ---------------------------------------------------------------------------
// Helper: collect system steps relevant to a round
// (system steps from the previous round + current round)
// ---------------------------------------------------------------------------

function collectSystemSteps(history: HistoryData, roundIndex: number): SystemStep[] {
  const result: SystemStep[] = [];
  if (roundIndex > 0) {
    const prev = history.rounds[roundIndex - 1]!;
    result.push(...(prev.steps.filter((s): s is SystemStep => s.type === "SystemStep")));
  }
  const cur = history.rounds[roundIndex]!;
  result.push(...(cur.steps.filter((s): s is SystemStep => s.type === "SystemStep")));
  return result;
}

// ---------------------------------------------------------------------------
// FullHistoryAgentMemory
// ---------------------------------------------------------------------------

export class FullHistoryAgentMemory extends AgentMemory {
  getMessages(history: HistoryData): ChatCompletionMessageParam[] {
    const messages: ChatCompletionMessageParam[] = [{ role: "system", content: this.systemPrompt }];

    for (let i = 0; i < history.rounds.length; i++) {
      const round = history.rounds[i]!;
      const sysCur = collectSystemSteps(history, i);

      for (const step of round.steps) {
        if (step.type === "UserStep") {
          messages.push(this.formatUserStep(step, sysCur));
        } else if (step.type === "SystemStep") {
          continue;
        } else if (step.type === "ObservationStep") {
          messages.push(this.formatObservationStep(step));
        } else if (step.type === "AgentStep") {
          messages.push(this.formatAgentStep(step));
        }
      }
    }
    return messages;
  }

  override getMemoryDebug(history: HistoryData): MemoryDebug {
    const debugMessages: DebugEntry[] = [];
    let idx = 0;

    const sysMsg: ChatCompletionMessageParam = { role: "system", content: this.systemPrompt };
    debugMessages.push(this.buildDebugEntry(idx++, sysMsg));

    for (let i = 0; i < history.rounds.length; i++) {
      const round = history.rounds[i]!;
      const sysCur = collectSystemSteps(history, i);

      for (const step of round.steps) {
        if (step.type === "UserStep") {
          debugMessages.push(this.buildDebugEntry(idx++, this.formatUserStep(step, sysCur), { step, roundNum: round.roundNum, stepType: "UserStep" }));
        } else if (step.type === "SystemStep") {
          continue;
        } else if (step.type === "ObservationStep") {
          debugMessages.push(this.buildDebugEntry(idx++, this.formatObservationStep(step), { step, roundNum: round.roundNum, stepType: "ObservationStep" }));
        } else if (step.type === "AgentStep") {
          debugMessages.push(this.buildDebugEntry(idx++, this.formatAgentStep(step), { step, roundNum: round.roundNum, stepType: "AgentStep" }));
        }
      }
    }

    return {
      memory_type: "FullHistoryAgentMemory",
      config: {},
      summary: {
        total_rounds: history.rounds.length,
        total_messages: debugMessages.length,
        total_content_length: debugMessages.reduce((s, m) => s + m.content_length, 0),
      },
      messages: debugMessages,
    };
  }

  onStepAdded(_step: Step): void {}
  async onStepAddedAsync(_step: Step): Promise<void> {}

  serialize(): Record<string, unknown> {
    return {};
  }
  deserialize(_data: Record<string, unknown>): void {}
  reset(): void {}
}

// ---------------------------------------------------------------------------
// SlidingWindowAgentMemory
// ---------------------------------------------------------------------------

export class SlidingWindowAgentMemory extends AgentMemory {
  constructor(
    public startWindowSize = 5,
    public endWindowSize = 20,
  ) {
    super();
  }

  private isFullRound(roundNum: number, totalRounds: number): boolean {
    return roundNum <= this.startWindowSize || roundNum > totalRounds - this.endWindowSize;
  }

  getMessages(history: HistoryData): ChatCompletionMessageParam[] {
    const messages: ChatCompletionMessageParam[] = [{ role: "system", content: this.systemPrompt }];
    const total = history.rounds.length;

    for (let i = 0; i < history.rounds.length; i++) {
      const round = history.rounds[i]!;
      const full = this.isFullRound(round.roundNum, total);
      const sysCur = collectSystemSteps(history, i);

      for (const step of round.steps) {
        if (step.type === "UserStep") {
          messages.push(this.formatUserStep(step, sysCur));
        } else if (full) {
          if (step.type === "ObservationStep") messages.push(this.formatObservationStep(step));
          else if (step.type === "AgentStep") messages.push(this.formatAgentStep(step));
        } else {
          // Middle round — only the final AgentStep
          if (step.type === "AgentStep" && step.isFinal) messages.push(this.formatAgentStep(step));
        }
      }
    }
    return messages;
  }

  override getMemoryDebug(history: HistoryData): MemoryDebug {
    const debugMessages: DebugEntry[] = [];
    let idx = 0;
    const total = history.rounds.length;
    const fullRounds: number[] = [];
    const windowedRounds: number[] = [];

    debugMessages.push(this.buildDebugEntry(idx++, { role: "system", content: this.systemPrompt }));

    for (let i = 0; i < history.rounds.length; i++) {
      const round = history.rounds[i]!;
      const full = this.isFullRound(round.roundNum, total);
      (full ? fullRounds : windowedRounds).push(round.roundNum);
      const sysCur = collectSystemSteps(history, i);

      for (const step of round.steps) {
        if (step.type === "UserStep") {
          debugMessages.push(this.buildDebugEntry(idx++, this.formatUserStep(step, sysCur), { step, roundNum: round.roundNum, stepType: "UserStep" }));
        } else if (full) {
          if (step.type === "ObservationStep") debugMessages.push(this.buildDebugEntry(idx++, this.formatObservationStep(step), { step, roundNum: round.roundNum, stepType: "ObservationStep" }));
          else if (step.type === "AgentStep") debugMessages.push(this.buildDebugEntry(idx++, this.formatAgentStep(step), { step, roundNum: round.roundNum, stepType: "AgentStep" }));
        } else {
          if (step.type === "AgentStep" && step.isFinal) debugMessages.push(this.buildDebugEntry(idx++, this.formatAgentStep(step), { step, roundNum: round.roundNum, stepType: "AgentStep" }));
        }
      }
    }

    return {
      memory_type: "SlidingWindowAgentMemory",
      config: { start_window_size: this.startWindowSize, end_window_size: this.endWindowSize },
      summary: {
        total_rounds: total,
        full_rounds: fullRounds,
        windowed_rounds: windowedRounds,
        total_messages: debugMessages.length,
        total_content_length: debugMessages.reduce((s, m) => s + m.content_length, 0),
      },
      messages: debugMessages,
    };
  }

  onStepAdded(_step: Step): void {}
  async onStepAddedAsync(_step: Step): Promise<void> {}

  serialize(): Record<string, unknown> {
    return { start_window_size: this.startWindowSize, end_window_size: this.endWindowSize };
  }
  deserialize(data: Record<string, unknown>): void {
    this.startWindowSize = data["start_window_size"] as number;
    this.endWindowSize = data["end_window_size"] as number;
  }
  reset(): void {}
}

// ---------------------------------------------------------------------------
// IndexedAgentMemory
// ---------------------------------------------------------------------------

const COMPRESSED_PREFIX = "[COMPRESSED MESSAGE]\n";

export class IndexedAgentMemory extends AgentMemory {
  private readonly llmEngine: LLMEngine;
  private compressThreshold: number;
  private recentWindowSize: number;
  private startWindowSize: number;
  private compressedCache: Map<string, string> = new Map();
  private pendingTasks: Set<Promise<void>> = new Set();

  private _compressTemplate?: string;
  private get compressTemplate(): string {
    return (this._compressTemplate ??= loadPrompt("IndexedMemory_compress_prompt_template"));
  }

  constructor(
    llmEngine: LLMEngine,
    opts?: {
      compressThreshold?: number;
      recentWindowSize?: number;
      startWindowSize?: number;
    },
  ) {
    super();
    this.llmEngine = llmEngine;
    this.compressThreshold = opts?.compressThreshold ?? 2000;
    this.recentWindowSize = opts?.recentWindowSize ?? 5;
    this.startWindowSize = opts?.startWindowSize ?? 3;
  }

  private getStepTypeLabel(step: Step): string {
    switch (step.type) {
      case "UserStep": return "user input";
      case "AgentStep": return "agent response";
      case "ObservationStep": return "tool observation";
      case "SystemStep": return "system message";
    }
  }

  private getFormattedContent(step: Step): string {
    switch (step.type) {
      case "UserStep": return (this.formatUserStep(step, []).content as string) ?? "";
      case "ObservationStep": return (this.formatObservationStep(step).content as string) ?? "";
      case "AgentStep": return (this.formatAgentStep(step).content as string) ?? "";
      case "SystemStep": return step.systemMessage;
    }
  }

  private async compressAsync(step: Step, content: string): Promise<void> {
    const typeLabel = this.getStepTypeLabel(step);
    const prompt = applyTemplate(this.compressTemplate, { type: typeLabel, content });
    const result = await this.llmEngine.chat([{ role: "user", content: prompt }], []);
    this.compressedCache.set(step.stepId, COMPRESSED_PREFIX + result.response);
  }

  onStepAdded(step: Step): void {
    // Sync compression is not supported — no-op (use onStepAddedAsync in the agent loop)
  }

  async onStepAddedAsync(step: Step): Promise<void> {
    const content = this.getFormattedContent(step);
    if (content.length > this.compressThreshold) {
      const task = this.compressAsync(step, content).catch((e) =>
        console.warn(`[IndexedAgentMemory] Failed to compress step ${step.stepId}:`, e),
      );
      this.pendingTasks.add(task);
      task.finally(() => this.pendingTasks.delete(task));
    }
  }

  private isFullRound(roundNum: number, totalRounds: number): boolean {
    return roundNum <= this.startWindowSize || roundNum > totalRounds - this.recentWindowSize;
  }

  getMessages(history: HistoryData): ChatCompletionMessageParam[] {
    const messages: ChatCompletionMessageParam[] = [{ role: "system", content: this.systemPrompt }];
    const total = history.rounds.length;

    for (let i = 0; i < history.rounds.length; i++) {
      const round = history.rounds[i]!;
      const full = this.isFullRound(round.roundNum, total);
      const sysCur = collectSystemSteps(history, i);

      for (const step of round.steps) {
        if (step.type === "UserStep") {
          messages.push(this.formatUserStep(step, sysCur));
        } else if (step.type === "SystemStep") {
          continue;
        } else if (full || !this.compressedCache.has(step.stepId)) {
          if (step.type === "ObservationStep") messages.push(this.formatObservationStep(step));
          else if (step.type === "AgentStep") messages.push(this.formatAgentStep(step));
        } else {
          const compressed = this.compressedCache.get(step.stepId)!;
          if (step.type === "ObservationStep") messages.push({ role: "user", content: compressed });
          else if (step.type === "AgentStep") messages.push({ role: "assistant", content: compressed });
        }
      }
    }
    return messages;
  }

  override getMemoryDebug(history: HistoryData): MemoryDebug {
    const debugMessages: DebugEntry[] = [];
    let idx = 0;
    const total = history.rounds.length;
    const fullRounds: number[] = [];
    const compressedRounds: number[] = [];
    let compressedStepsCount = 0;

    debugMessages.push(this.buildDebugEntry(idx++, { role: "system", content: this.systemPrompt }));

    for (let i = 0; i < history.rounds.length; i++) {
      const round = history.rounds[i]!;
      const full = this.isFullRound(round.roundNum, total);
      (full ? fullRounds : compressedRounds).push(round.roundNum);
      const sysCur = collectSystemSteps(history, i);

      for (const step of round.steps) {
        if (step.type === "UserStep") {
          debugMessages.push(this.buildDebugEntry(idx++, this.formatUserStep(step, sysCur), { step, roundNum: round.roundNum, stepType: "UserStep" }));
        } else if (step.type === "SystemStep") {
          continue;
        } else if (full || !this.compressedCache.has(step.stepId)) {
          if (step.type === "ObservationStep") debugMessages.push(this.buildDebugEntry(idx++, this.formatObservationStep(step), { step, roundNum: round.roundNum, stepType: "ObservationStep" }));
          else if (step.type === "AgentStep") debugMessages.push(this.buildDebugEntry(idx++, this.formatAgentStep(step), { step, roundNum: round.roundNum, stepType: "AgentStep" }));
        } else {
          const compressed = this.compressedCache.get(step.stepId)!;
          compressedStepsCount++;
          if (step.type === "ObservationStep") {
            const orig = this.formatObservationStep(step);
            debugMessages.push(this.buildDebugEntry(idx++, { role: "user", content: compressed }, { step, roundNum: round.roundNum, stepType: "ObservationStep", isCompressed: true, originalContentLength: (orig.content as string).length }));
          } else if (step.type === "AgentStep") {
            const orig = this.formatAgentStep(step);
            debugMessages.push(this.buildDebugEntry(idx++, { role: "assistant", content: compressed }, { step, roundNum: round.roundNum, stepType: "AgentStep", isCompressed: true, originalContentLength: (orig.content as string).length }));
          }
        }
      }
    }

    return {
      memory_type: "IndexedAgentMemory",
      config: { start_window_size: this.startWindowSize, recent_window_size: this.recentWindowSize, compress_threshold: this.compressThreshold },
      summary: {
        total_rounds: total,
        full_rounds: fullRounds,
        compressed_rounds: compressedRounds,
        total_messages: debugMessages.length,
        compressed_steps: compressedStepsCount,
        pending_compressions: this.pendingTasks.size,
        total_content_length: debugMessages.reduce((s, m) => s + m.content_length, 0),
      },
      messages: debugMessages,
    };
  }

  override getCompressionDebug(history: HistoryData): CompressionDebug {
    const entries: CompressionDebug["entries"] = [];
    for (const round of history.rounds) {
      for (const step of round.steps) {
        if (this.compressedCache.has(step.stepId)) {
          const original = this.getFormattedContent(step);
          const compressed = this.compressedCache.get(step.stepId)!;
          entries.push({
            step_id: step.stepId,
            step_type: step.type,
            round_num: round.roundNum,
            original_content: original,
            compressed_content: compressed,
            original_length: original.length,
            compressed_length: compressed.length,
            compression_ratio: original.length > 0 ? Math.round((compressed.length / original.length) * 10000) / 10000 : null,
          });
        }
      }
    }
    return {
      memory_type: "IndexedAgentMemory",
      compression_supported: true,
      config: { compress_threshold: this.compressThreshold },
      total_compressed: this.compressedCache.size,
      pending_compressions: this.pendingTasks.size,
      entries,
    };
  }

  serialize(): Record<string, unknown> {
    return {
      compressed_cache: Object.fromEntries(this.compressedCache),
      compress_threshold: this.compressThreshold,
      recent_window_size: this.recentWindowSize,
      start_window_size: this.startWindowSize,
    };
  }

  deserialize(data: Record<string, unknown>): void {
    const cache = (data["compressed_cache"] ?? {}) as Record<string, string>;
    this.compressedCache = new Map(Object.entries(cache));
    if (data["compress_threshold"] != null) this.compressThreshold = data["compress_threshold"] as number;
    if (data["recent_window_size"] != null) this.recentWindowSize = data["recent_window_size"] as number;
    if (data["start_window_size"] != null) this.startWindowSize = data["start_window_size"] as number;
  }

  reset(): void {
    this.compressedCache.clear();
    this.pendingTasks.clear();
  }
}
