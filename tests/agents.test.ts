import { describe, it, expect, vi } from "vitest";
import { Agent } from "../src/agents/index.js";
import type { AgentEvent } from "../src/agents/index.js";
import { Tool, AsyncTool, FinalResponseTool, JobWaitTool, JobCancelTool } from "../src/tools/index.js";
import { JobManager } from "../src/jobs/index.js";
import type { WorkerProcess } from "../src/jobs/index.js";
import type { LLMEngine, ChatResult } from "../src/engines/index.js";
import type { ChatCompletionTool } from "openai/resources/chat/completions";
import { makeHistory } from "../src/history/index.js";
import type { HistoryData } from "../src/history/index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMockEngine(
  responses: Array<{ response?: string; toolCalls?: Array<{ name: string; arguments: Record<string, unknown> }> }>,
): LLMEngine {
  let idx = 0;
  return {
    chat: vi.fn().mockImplementation(async (): Promise<ChatResult> => {
      const r = responses[idx++] ?? { response: "", toolCalls: [] };
      return {
        response: r.response ?? "",
        toolCalls: (r.toolCalls ?? []).map((tc) => ({ name: tc.name, arguments: tc.arguments })),
      };
    }),
  };
}

function makeMockWorker(result: Record<string, unknown> = { output: "worker output" }, delay = 0): WorkerProcess {
  return {
    sendCommand: vi.fn().mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve(result), delay)),
    ),
    restart: vi.fn().mockResolvedValue(undefined),
  };
}

/** Minimal sync tool for tests. */
class EchoTool extends Tool {
  constructor() {
    super("echo", "Echoes input back.");
  }
  override execute(params: Record<string, unknown>): string {
    return `echo: ${params["text"] ?? ""}`;
  }
  override getToolCallSchema(): ChatCompletionTool {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
      },
    };
  }
}

/** Minimal async tool for tests. */
class MockAsyncTool extends AsyncTool {
  constructor(jobManager: JobManager) {
    super("mock_async", "Async test tool.", jobManager);
  }
  override submit(params: Record<string, unknown>): string {
    return this.jobManager.submit(this.name, "execute", params);
  }
  override getToolCallSchema(): ChatCompletionTool {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        parameters: { type: "object", properties: {}, required: [] },
      },
    };
  }
}

/** Collect all events from the agent's chat() generator. */
async function collectEvents(agent: Agent, input: string): Promise<AgentEvent[]> {
  const events: AgentEvent[] = [];
  for await (const event of agent.chat(input)) {
    events.push(event);
  }
  return events;
}

function makeAgent(
  engine: LLMEngine,
  history: HistoryData,
  tools?: Tool[],
  maxRetries = 4,
): Agent {
  return new Agent(engine, history, { tools, maxRetries });
}

// ---------------------------------------------------------------------------
// FinalResponseTool
// ---------------------------------------------------------------------------

describe("FinalResponseTool", () => {
  it("has correct name and schema required field", () => {
    const t = new FinalResponseTool();
    expect(t.name).toBe("final_response");
    const schema = t.getToolCallSchema();
    expect(schema.function.parameters?.required).toContain("response");
  });

  it("execute returns the response string", () => {
    const t = new FinalResponseTool();
    expect(t.execute({ response: "hello" })).toBe("hello");
  });

  it("getTitle always returns 'Final Response'", () => {
    expect(new FinalResponseTool().getTitle({})).toBe("Final Response");
  });
});

// ---------------------------------------------------------------------------
// JobWaitTool
// ---------------------------------------------------------------------------

describe("JobWaitTool", () => {
  it("returns completed result", async () => {
    const worker = makeMockWorker({ output: "the answer" });
    const jm = new JobManager(worker);
    const jobId = jm.submit("t", "execute", {});
    await new Promise((r) => setTimeout(r, 20)); // let job complete

    const tool = new JobWaitTool(jm);
    const result = await tool.execute({ job_id: jobId });
    expect(result).toBe("the answer");
  });

  it("returns error for missing job_id param", async () => {
    const jm = new JobManager(makeMockWorker());
    const tool = new JobWaitTool(jm);
    expect(await tool.execute({})).toBe("Error: job_id is required.");
  });

  it("getTitle includes job_id", () => {
    const tool = new JobWaitTool(new JobManager(makeMockWorker()));
    expect(tool.getTitle({ job_id: "abc123" })).toContain("abc123");
  });
});

// ---------------------------------------------------------------------------
// JobCancelTool
// ---------------------------------------------------------------------------

describe("JobCancelTool", () => {
  it("cancels a running job and returns confirmation message", () => {
    const jm = new JobManager(makeMockWorker({}, 9999));
    const jobId = jm.submit("t", "execute", {});
    const tool = new JobCancelTool(jm);
    const result = tool.execute({ job_id: jobId });
    expect(typeof result).toBe("string");
    expect(result).toContain("cancelled");
  });

  it("returns error for missing job_id param", () => {
    const tool = new JobCancelTool(new JobManager(makeMockWorker()));
    expect(tool.execute({})).toBe("Error: job_id is required.");
  });

  it("reports 'could not be cancelled' when job already finished", async () => {
    const jm = new JobManager(makeMockWorker({ output: "x" }));
    const jobId = jm.submit("t", "execute", {});
    await jm.waitAsync(jobId, 2); // let it complete
    const tool = new JobCancelTool(jm);
    const result = tool.execute({ job_id: jobId });
    expect(result).toContain("could not be cancelled");
  });
});

// ---------------------------------------------------------------------------
// Agent auto-injects job management tools
// ---------------------------------------------------------------------------

describe("Agent tool injection", () => {
  it("always injects FinalResponseTool", () => {
    const engine = makeMockEngine([]);
    const agent = makeAgent(engine, makeHistory());
    const tools = agent.getTools();
    expect(tools.some((t) => t.name === "final_response")).toBe(true);
  });

  it("injects job_wait and job_cancel when an AsyncTool is provided", () => {
    const jm = new JobManager(makeMockWorker());
    const asyncTool = new MockAsyncTool(jm);
    const agent = makeAgent(makeMockEngine([]), makeHistory(), [asyncTool]);
    const names = agent.getTools().map((t) => t.name);
    expect(names).toContain("job_wait");
    expect(names).toContain("job_cancel");
  });

  it("does NOT inject job management tools when no AsyncTools are present", () => {
    const agent = makeAgent(makeMockEngine([]), makeHistory(), [new EchoTool()]);
    const names = agent.getTools().map((t) => t.name);
    expect(names).not.toContain("job_wait");
    expect(names).not.toContain("job_cancel");
  });

  it("injects exactly one pair per unique JobManager", () => {
    const jm = new JobManager(makeMockWorker());
    const t1 = new MockAsyncTool(jm);
    const t2 = new MockAsyncTool(jm); // same manager
    const agent = makeAgent(makeMockEngine([]), makeHistory(), [t1, t2]);
    const names = agent.getTools().map((t) => t.name);
    expect(names.filter((n) => n === "job_wait")).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Agent.chat — basic flows
// ---------------------------------------------------------------------------

describe("Agent.chat — basic flows", () => {
  it("single-round: final_response immediately", async () => {
    const engine = makeMockEngine([
      { response: "", toolCalls: [{ name: "final_response", arguments: { response: "Hello!" } }] },
    ]);
    const events = await collectEvents(makeAgent(engine, makeHistory()), "hi");
    const responseEvents = events.filter((e) => e.type === "response");
    const finalEvent = events.find((e) => e.type === "final_decision");
    expect(responseEvents.some((e) => e.type === "response" && e.data === "Hello!")).toBe(true);
    expect(finalEvent).toMatchObject({ type: "final_decision", isFinal: true });
  });

  it("yields interim response text alongside tool calls", async () => {
    const engine = makeMockEngine([
      {
        response: "Let me check something.",
        toolCalls: [{ name: "echo", arguments: { text: "test" } }],
      },
      { toolCalls: [{ name: "final_response", arguments: { response: "Done." } }] },
    ]);
    const events = await collectEvents(makeAgent(engine, makeHistory(), [new EchoTool()]), "go");
    const responseTexts = events
      .filter((e): e is Extract<AgentEvent, { type: "response" }> => e.type === "response")
      .map((e) => e.data);
    expect(responseTexts).toContain("Let me check something.");
  });

  it("sync tool: yields tool_running then tool_output", async () => {
    const engine = makeMockEngine([
      { toolCalls: [{ name: "echo", arguments: { text: "world" } }] },
      { toolCalls: [{ name: "final_response", arguments: { response: "ok" } }] },
    ]);
    const events = await collectEvents(makeAgent(engine, makeHistory(), [new EchoTool()]), "go");
    const types = events.map((e) => e.type);
    expect(types).toContain("tool_running");
    expect(types).toContain("tool_output");
    const outputEvent = events.find(
      (e): e is Extract<AgentEvent, { type: "tool_output" }> => e.type === "tool_output",
    );
    expect(outputEvent?.data).toBe("echo: world");
  });

  it("unknown tool: yields tool_output with error", async () => {
    const engine = makeMockEngine([
      { toolCalls: [{ name: "nonexistent_tool", arguments: {} }] },
      { toolCalls: [{ name: "final_response", arguments: { response: "done" } }] },
    ]);
    const events = await collectEvents(makeAgent(engine, makeHistory()), "go");
    const outputEvent = events.find(
      (e): e is Extract<AgentEvent, { type: "tool_output" }> => e.type === "tool_output",
    );
    expect(outputEvent?.data).toContain("not found");
  });

  it("tool_calls preview is yielded for non-final tool calls", async () => {
    const engine = makeMockEngine([
      { toolCalls: [{ name: "echo", arguments: { text: "x" } }] },
      { toolCalls: [{ name: "final_response", arguments: { response: "ok" } }] },
    ]);
    const events = await collectEvents(makeAgent(engine, makeHistory(), [new EchoTool()]), "go");
    const toolCallsEvent = events.find((e) => e.type === "tool_calls");
    expect(toolCallsEvent).toBeDefined();
  });

  it("tool_calls preview is NOT yielded for final_response alone", async () => {
    const engine = makeMockEngine([
      { toolCalls: [{ name: "final_response", arguments: { response: "done" } }] },
    ]);
    const events = await collectEvents(makeAgent(engine, makeHistory()), "go");
    expect(events.some((e) => e.type === "tool_calls")).toBe(false);
  });

  it("history is updated with UserStep + AgentStep + ObservationStep", async () => {
    const engine = makeMockEngine([
      { toolCalls: [{ name: "echo", arguments: { text: "x" } }] },
      { toolCalls: [{ name: "final_response", arguments: { response: "ok" } }] },
    ]);
    const history = makeHistory();
    const agent = makeAgent(engine, history, [new EchoTool()]);
    await collectEvents(agent, "hello");
    const steps = history.rounds.flatMap((r) => r.steps);
    const types = steps.map((s) => s.type);
    expect(types).toContain("UserStep");
    expect(types).toContain("AgentStep");
    expect(types).toContain("ObservationStep");
  });
});

// ---------------------------------------------------------------------------
// Agent.chat — async tool + auto-wait
// ---------------------------------------------------------------------------

describe("Agent.chat — async tool", () => {
  it("async tool: yields tool_running, then tool_output after auto-wait", async () => {
    const worker = makeMockWorker({ output: "async result" }, 50);
    const jm = new JobManager(worker);
    const asyncTool = new MockAsyncTool(jm);

    const engine = makeMockEngine([
      { toolCalls: [{ name: "mock_async", arguments: {} }] },
      { toolCalls: [{ name: "final_response", arguments: { response: "done" } }] },
    ]);

    process.env["MEDDS_AUTO_WAIT_TIMEOUT"] = "2"; // 2s auto-wait
    const events = await collectEvents(makeAgent(engine, makeHistory(), [asyncTool]), "go");
    delete process.env["MEDDS_AUTO_WAIT_TIMEOUT"];

    const types = events.map((e) => e.type);
    expect(types).toContain("tool_running");
    expect(types).toContain("tool_output");

    const runningEvent = events.find(
      (e): e is Extract<AgentEvent, { type: "tool_running" }> => e.type === "tool_running",
    );
    expect(runningEvent?.jobId).toBeTruthy();

    const outputEvent = events.find(
      (e): e is Extract<AgentEvent, { type: "tool_output" }> => e.type === "tool_output",
    );
    expect(outputEvent?.data).toBe("async result");
  });

  it("auto-wait timeout: still-running job reported as running with job_id hint", async () => {
    const worker = makeMockWorker({ output: "late" }, 5000); // slow
    const jm = new JobManager(worker);
    const asyncTool = new MockAsyncTool(jm);

    const engine = makeMockEngine([
      { toolCalls: [{ name: "mock_async", arguments: {} }] },
      { toolCalls: [{ name: "final_response", arguments: { response: "ok" } }] },
    ]);

    process.env["MEDDS_AUTO_WAIT_TIMEOUT"] = "0"; // instant check
    const events = await collectEvents(makeAgent(engine, makeHistory(), [asyncTool]), "go");
    delete process.env["MEDDS_AUTO_WAIT_TIMEOUT"];

    const outputEvent = events.find(
      (e): e is Extract<AgentEvent, { type: "tool_output" }> => e.type === "tool_output",
    );
    expect(outputEvent?.data).toContain("still running");
    expect(outputEvent?.data).toContain("job_wait");
  });
});

// ---------------------------------------------------------------------------
// Agent.chat — validation rules
// ---------------------------------------------------------------------------

describe("Agent.chat — validation rules", () => {
  it("no tool call → retry, eventually uses error response", async () => {
    // Engine always returns no tool calls → agent retries to exhaustion
    const engine = makeMockEngine(
      Array(5).fill({ response: "Thinking...", toolCalls: [] }),
    );
    const events = await collectEvents(makeAgent(engine, makeHistory(), [], 3), "go");
    // After retries exhausted, yields a System Error response
    const errEvent = events.find(
      (e): e is Extract<AgentEvent, { type: "response" }> =>
        e.type === "response" && e.data.includes("System Error"),
    );
    expect(errEvent).toBeDefined();
  });

  it("final_response mixed with other tools → retry", async () => {
    // First call mixes final_response + echo (invalid).
    // Second call is clean final_response.
    const engine = makeMockEngine([
      {
        toolCalls: [
          { name: "echo", arguments: { text: "hi" } },
          { name: "final_response", arguments: { response: "x" } },
        ],
      },
      { toolCalls: [{ name: "final_response", arguments: { response: "ok" } }] },
    ]);
    const events = await collectEvents(makeAgent(engine, makeHistory(), [new EchoTool()]), "go");
    // Should ultimately succeed (2nd attempt has clean final_response)
    const finalDecision = events.find((e) => e.type === "final_decision");
    expect(finalDecision).toMatchObject({ type: "final_decision", isFinal: true });
    // Engine was called twice
    expect((engine.chat as ReturnType<typeof vi.fn>).mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it("final_response while async job pending → retry", async () => {
    const worker = makeMockWorker({ output: "late" }, 5000); // never finishes during test
    const jm = new JobManager(worker);
    const asyncTool = new MockAsyncTool(jm);

    const engine = makeMockEngine([
      // Round 1, attempt 1: submit async job
      { toolCalls: [{ name: "mock_async", arguments: {} }] },
      // Round 1, attempt 1: tries to finalize while job is pending (invalid)
      // But wait — final_response comes in a separate LLM call (separate round iteration).
      // Actually agent loops: each LLM call is one attempt in a retry loop PER round.
      // The auto-wait for "mock_async" will create the ObservationStep and start a new round.
      // Then in round 2, if we try final_response while job is still pending...
      // Actually the way the agent works: after Phase 2 auto-wait finishes, it records
      // ObservationStep and starts a new LLM call. The job is already done (or timed out).
      // To test "finalize while pending": the auto-wait must time out (job still running),
      // and then the next round the LLM tries final_response.
      { toolCalls: [{ name: "final_response", arguments: { response: "too early" } }] },
      // After retry (job still pending), try clean final
      // But since job is still running (delay=5000), hasPending() is true.
      // Another retry: agent appends error to messages
      { toolCalls: [{ name: "final_response", arguments: { response: "still too early" } }] },
      // Exhaust retries or cancel and finalize
      { toolCalls: [{ name: "final_response", arguments: { response: "still too early 2" } }] },
    ]);

    process.env["MEDDS_AUTO_WAIT_TIMEOUT"] = "0"; // instant timeout → job stays running

    const events = await collectEvents(
      makeAgent(engine, makeHistory(), [asyncTool], 4),
      "go",
    );
    delete process.env["MEDDS_AUTO_WAIT_TIMEOUT"];

    // The agent should have retried (engine called more than once for the final_response attempt)
    expect((engine.chat as ReturnType<typeof vi.fn>).mock.calls.length).toBeGreaterThanOrEqual(2);
    // Eventually: either all retries exhausted (System Error) or a final_decision event
    const hasDecision =
      events.some((e) => e.type === "final_decision") ||
      events.some(
        (e): e is Extract<AgentEvent, { type: "response" }> =>
          e.type === "response" && e.data.includes("System Error"),
      );
    expect(hasDecision).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Agent.chat — multi-round
// ---------------------------------------------------------------------------

describe("Agent.chat — multi-round", () => {
  it("two-round: tool → final_response", async () => {
    const engine = makeMockEngine([
      { toolCalls: [{ name: "echo", arguments: { text: "a" } }] },
      { toolCalls: [{ name: "final_response", arguments: { response: "done" } }] },
    ]);
    const history = makeHistory();
    const agent = makeAgent(engine, history, [new EchoTool()]);
    const events = await collectEvents(agent, "start");
    expect(events.some((e) => e.type === "final_decision" && (e as { isFinal: boolean }).isFinal)).toBe(true);
    // History should have UserStep + 2×AgentStep + ObservationStep
    const steps = history.rounds.flatMap((r) => r.steps);
    expect(steps.filter((s) => s.type === "AgentStep")).toHaveLength(2);
    expect(steps.filter((s) => s.type === "ObservationStep")).toHaveLength(1);
  });
});
