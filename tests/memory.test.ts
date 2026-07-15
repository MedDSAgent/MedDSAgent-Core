import { describe, it, expect, vi } from "vitest";
import {
  FullHistoryAgentMemory,
  SlidingWindowAgentMemory,
  IndexedAgentMemory,
} from "../src/memory/index.js";
import {
  makeHistory,
  addStep,
  makeUserStep,
  makeAgentStep,
  makeObservationStep,
  makeSystemStep,
} from "../src/history/index.js";
import type { HistoryData } from "../src/history/index.js";
import type { LLMEngine } from "../src/engines/index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a history with N complete user→agent rounds. */
function buildHistory(rounds: number, includeSystemStep = false): HistoryData {
  const h = makeHistory();
  if (includeSystemStep) {
    addStep(h, makeSystemStep("db connected"));
  }
  for (let r = 1; r <= rounds; r++) {
    addStep(h, makeUserStep(`question ${r}`));
    addStep(h, makeAgentStep("a1", { tools: [{ tool_name: "PythonExecutor", tool_args: `{"code":"print(${r})"}`, tool_title: "" }] }));
    addStep(h, makeObservationStep("a1", [{ tool_name: "PythonExecutor", output: `${r}` }]));
    addStep(h, makeAgentStep("a1", { tools: [{ tool_name: "final_response", tool_args: `{"response":"answer ${r}"}`, tool_title: "" }], isFinal: true }));
  }
  return h;
}

// ---------------------------------------------------------------------------
// FullHistoryAgentMemory
// ---------------------------------------------------------------------------

describe("FullHistoryAgentMemory", () => {
  it("first message is always the system prompt", () => {
    const mem = new FullHistoryAgentMemory();
    mem.setSystemPrompt("You are a data scientist.");
    const msgs = mem.getMessages(buildHistory(1));
    expect(msgs[0]).toEqual({ role: "system", content: "You are a data scientist." });
  });

  it("specialty prompt is appended to system prompt", () => {
    const mem = new FullHistoryAgentMemory();
    mem.setSystemPrompt("Base.", "Specialty info.");
    const msgs = mem.getMessages(buildHistory(1));
    expect((msgs[0]!.content as string)).toContain("Specialty info.");
  });

  it("produces role sequence: system, user, assistant, user (obs), assistant (final)", () => {
    const mem = new FullHistoryAgentMemory();
    mem.setSystemPrompt("sys");
    const msgs = mem.getMessages(buildHistory(1));
    expect(msgs.map((m) => m.role)).toEqual(["system", "user", "assistant", "user", "assistant"]);
  });

  it("keeps all rounds for N-round history", () => {
    const mem = new FullHistoryAgentMemory();
    mem.setSystemPrompt("sys");
    const msgs = mem.getMessages(buildHistory(3));
    // system + 3 × (user + assistant + user + assistant) = 1 + 12 = 13
    expect(msgs).toHaveLength(13);
  });

  it("SystemStep message is embedded in the following UserStep, not standalone", () => {
    const mem = new FullHistoryAgentMemory();
    mem.setSystemPrompt("sys");
    const h = buildHistory(1, true);
    const msgs = mem.getMessages(h);
    // System step goes into round 0 but should NOT appear as a standalone message
    const roles = msgs.map((m) => m.role);
    expect(roles.filter((r) => r === "system")).toHaveLength(1); // only the main system prompt
    // The SystemStep content should be embedded in the user message
    const userMsg = msgs.find((m) => m.role === "user");
    expect((userMsg!.content as string)).toContain("db connected");
  });

  it("serialize/deserialize is a no-op (stateless)", () => {
    const mem = new FullHistoryAgentMemory();
    mem.deserialize(mem.serialize());
    expect(mem.serialize()).toEqual({});
  });

  it("getMemoryDebug returns correct memory_type and message count", () => {
    const mem = new FullHistoryAgentMemory();
    mem.setSystemPrompt("sys");
    const debug = mem.getMemoryDebug(buildHistory(2));
    expect(debug.memory_type).toBe("FullHistoryAgentMemory");
    // system + 2 × (user + assistant + user(obs) + assistant(final)) = 1 + 8 = 9
    expect(debug.messages).toHaveLength(9);
  });
});

// ---------------------------------------------------------------------------
// SlidingWindowAgentMemory
// ---------------------------------------------------------------------------

describe("SlidingWindowAgentMemory", () => {
  it("keeps all steps for rounds within the windows", () => {
    // start=2, end=2 — with only 3 rounds, all are covered by one window or the other
    const mem = new SlidingWindowAgentMemory(2, 2);
    mem.setSystemPrompt("sys");
    const msgs = mem.getMessages(buildHistory(3));
    expect(msgs.map((m) => m.role)).toEqual(["system", "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]);
  });

  it("middle rounds only emit UserStep + final AgentStep", () => {
    // start=1, end=1 — round 2 is the middle round (of 3)
    const mem = new SlidingWindowAgentMemory(1, 1);
    mem.setSystemPrompt("sys");
    const msgs = mem.getMessages(buildHistory(3));
    const roles = msgs.map((m) => m.role);
    // system + round1(user,assistant,user,assistant) + round2(user,assistant) + round3(user,assistant,user,assistant)
    expect(roles).toEqual([
      "system",
      "user", "assistant", "user", "assistant", // round 1 full
      "user", "assistant",                        // round 2 windowed (only UserStep + final)
      "user", "assistant", "user", "assistant",  // round 3 full
    ]);
  });

  it("serialize/deserialize preserves window sizes", () => {
    const mem = new SlidingWindowAgentMemory(7, 15);
    const mem2 = new SlidingWindowAgentMemory();
    mem2.deserialize(mem.serialize());
    expect(mem2.startWindowSize).toBe(7);
    expect(mem2.endWindowSize).toBe(15);
  });

  it("getMemoryDebug summary includes full_rounds and windowed_rounds", () => {
    const mem = new SlidingWindowAgentMemory(1, 1);
    mem.setSystemPrompt("sys");
    const debug = mem.getMemoryDebug(buildHistory(3));
    const summary = debug.summary as Record<string, unknown>;
    expect(summary["full_rounds"]).toContain(1);
    expect(summary["full_rounds"]).toContain(3);
    expect(summary["windowed_rounds"]).toContain(2);
  });

  it("reset is a no-op (stateless)", () => {
    const mem = new SlidingWindowAgentMemory();
    expect(() => mem.reset()).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// IndexedAgentMemory
// ---------------------------------------------------------------------------

describe("IndexedAgentMemory", () => {
  function makeMockEngine(response = "compressed summary"): LLMEngine {
    return {
      chat: vi.fn().mockResolvedValue({ response, toolCalls: [] }),
    };
  }

  it("below threshold: no compression, uses full content", () => {
    const engine = makeMockEngine();
    const mem = new IndexedAgentMemory(engine, { compressThreshold: 100000 });
    mem.setSystemPrompt("sys");
    const msgs = mem.getMessages(buildHistory(1));
    expect(msgs.map((m) => m.role)).toEqual(["system", "user", "assistant", "user", "assistant"]);
  });

  it("compressed steps appear in middle rounds", async () => {
    const engine = makeMockEngine("short summary");
    const mem = new IndexedAgentMemory(engine, {
      compressThreshold: 10, // very low — everything gets compressed
      startWindowSize: 1,
      recentWindowSize: 1,
    });
    mem.setSystemPrompt("sys");
    const h = buildHistory(3);

    // Trigger async compression for all steps
    for (const round of h.rounds) {
      for (const step of round.steps) {
        await mem.onStepAddedAsync(step);
      }
    }
    // Wait for all pending tasks
    await Promise.allSettled([...Array.from({ length: 10 }).map(() => Promise.resolve())]);

    const msgs = mem.getMessages(h);
    const middleRoundMsgs = msgs.slice(5, 7); // round 2 messages
    const hasCompressed = middleRoundMsgs.some(
      (m) => typeof m.content === "string" && m.content.startsWith("[COMPRESSED MESSAGE]"),
    );
    expect(hasCompressed).toBe(true);
  });

  it("onStepAdded (sync) is a no-op — does not compress", () => {
    const engine = makeMockEngine();
    const mem = new IndexedAgentMemory(engine, { compressThreshold: 0 });
    const step = makeUserStep("x");
    expect(() => mem.onStepAdded(step)).not.toThrow();
    expect(engine.chat).not.toHaveBeenCalled();
  });

  it("compress threshold: step below threshold not compressed", async () => {
    const engine = makeMockEngine();
    const mem = new IndexedAgentMemory(engine, { compressThreshold: 100000 });
    mem.setSystemPrompt("sys");
    const step = makeUserStep("short");
    await mem.onStepAddedAsync(step);
    expect(engine.chat).not.toHaveBeenCalled();
  });

  it("compress threshold: step above threshold triggers compression", async () => {
    const engine = makeMockEngine("summary");
    const mem = new IndexedAgentMemory(engine, { compressThreshold: 1 });
    const step = makeObservationStep("a1", [{ tool_name: "PythonExecutor", output: "x".repeat(100) }]);
    await mem.onStepAddedAsync(step);
    await new Promise((r) => setTimeout(r, 50));
    expect(engine.chat).toHaveBeenCalledOnce();
  });

  it("serialize/deserialize preserves compressed cache", () => {
    const engine = makeMockEngine();
    const mem = new IndexedAgentMemory(engine, { compressThreshold: 2000, startWindowSize: 3, recentWindowSize: 5 });
    const raw = mem.serialize();
    const mem2 = new IndexedAgentMemory(engine);
    mem2.deserialize(raw);
    expect(mem2.serialize()).toEqual(raw);
  });

  it("reset clears the compressed cache", async () => {
    const engine = makeMockEngine("s");
    const mem = new IndexedAgentMemory(engine, { compressThreshold: 1 });
    const step = makeUserStep("x".repeat(100));
    await mem.onStepAddedAsync(step);
    await new Promise((r) => setTimeout(r, 50));
    mem.reset();
    expect(mem.serialize()["compressed_cache"]).toEqual({});
  });

  it("getCompressionDebug returns compression_supported: true", () => {
    const mem = new IndexedAgentMemory(makeMockEngine());
    expect(mem.getCompressionDebug(buildHistory(1)).compression_supported).toBe(true);
  });
});
