import { describe, it, expect } from "vitest";
import {
  makeHistory,
  addStep,
  makeUserStep,
  makeAgentStep,
  makeObservationStep,
  makeSystemStep,
  serializeHistory,
  deserializeHistory,
  serializeStep,
  deserializeStep,
  getUserStep,
  getAnswerStep,
} from "../src/history/index.js";

describe("step serialization round-trip", () => {
  it("SystemStep", () => {
    const s = makeSystemStep("db connected");
    expect(deserializeStep(serializeStep(s))).toEqual(s);
  });

  it("UserStep", () => {
    const s = makeUserStep("hello agent");
    expect(deserializeStep(serializeStep(s))).toEqual(s);
  });

  it("AgentStep with tools", () => {
    const s = makeAgentStep("agent-1", {
      response: "let me run this",
      tools: [{ tool_name: "PythonExecutor", tool_args: '{"code":"print(1)"}', tool_title: "Run Python" }],
      isFinal: false,
    });
    expect(deserializeStep(serializeStep(s))).toEqual(s);
  });

  it("AgentStep final", () => {
    const s = makeAgentStep("agent-1", {
      tools: [{ tool_name: "final_response", tool_args: '{"response":"done"}', tool_title: "" }],
      isFinal: true,
    });
    const rt = deserializeStep(serializeStep(s)) as typeof s;
    expect(rt.isFinal).toBe(true);
  });

  it("ObservationStep", () => {
    const s = makeObservationStep("agent-1", [{ tool_name: "PythonExecutor", output: "2" }]);
    expect(deserializeStep(serializeStep(s))).toEqual(s);
  });
});

describe("History.addStep", () => {
  it("UserStep starts a new round", () => {
    const h = makeHistory();
    addStep(h, makeUserStep("hi"));
    expect(h.rounds).toHaveLength(1);
    expect(h.currentRoundNum).toBe(1);
    expect(h.rounds[0]?.steps[0]?.type).toBe("UserStep");
  });

  it("SystemStep before any UserStep goes into round 0", () => {
    const h = makeHistory();
    addStep(h, makeSystemStep("system event"));
    expect(h.rounds[0]?.steps[0]?.type).toBe("SystemStep");
    expect(h.rounds[0]?.roundNum).toBe(0);
  });

  it("non-UserStep before UserStep throws", () => {
    const h = makeHistory();
    expect(() => addStep(h, makeAgentStep("a"))).toThrow();
  });

  it("multiple rounds increment roundNum", () => {
    const h = makeHistory();
    addStep(h, makeUserStep("first"));
    addStep(h, makeAgentStep("a", { isFinal: true }));
    addStep(h, makeUserStep("second"));
    expect(h.rounds).toHaveLength(2);
    expect(h.currentRoundNum).toBe(2);
  });

  it("non-UserStep goes into the current round", () => {
    const h = makeHistory();
    addStep(h, makeUserStep("ask"));
    addStep(h, makeObservationStep("a", [{ tool_name: "PythonExecutor", output: "2" }]));
    expect(h.rounds[0]?.steps).toHaveLength(2);
  });
});

describe("History serialization round-trip", () => {
  it("full conversation round-trips through JSON", () => {
    const h = makeHistory();
    addStep(h, makeSystemStep("connected"));
    addStep(h, makeUserStep("what is 1+1?"));
    addStep(
      h,
      makeAgentStep("agent-1", {
        tools: [{ tool_name: "PythonExecutor", tool_args: '{"code":"print(1+1)"}', tool_title: "Run" }],
      }),
    );
    addStep(h, makeObservationStep("agent-1", [{ tool_name: "PythonExecutor", output: "2" }]));
    addStep(
      h,
      makeAgentStep("agent-1", {
        tools: [{ tool_name: "final_response", tool_args: '{"response":"The answer is 2."}', tool_title: "" }],
        isFinal: true,
      }),
    );

    const raw = JSON.parse(JSON.stringify(serializeHistory(h)));
    const h2 = deserializeHistory(raw);
    expect(h2.rounds).toHaveLength(h.rounds.length);
    expect(h2.currentRoundNum).toBe(h.currentRoundNum);
    expect(h2.rounds.map((r) => r.steps.map((s) => s.type))).toEqual(
      h.rounds.map((r) => r.steps.map((s) => s.type)),
    );
  });
});

describe("Round helpers", () => {
  it("getUserStep finds the UserStep", () => {
    const h = makeHistory();
    addStep(h, makeUserStep("hello"));
    const round = h.rounds[0]!;
    expect(getUserStep(round).userInput).toBe("hello");
  });

  it("getAnswerStep finds the final AgentStep", () => {
    const h = makeHistory();
    addStep(h, makeUserStep("q"));
    addStep(h, makeAgentStep("a", { isFinal: true, tools: [{ tool_name: "final_response", tool_args: "{}", tool_title: "" }] }));
    const round = h.rounds[0]!;
    expect(getAnswerStep(round).isFinal).toBe(true);
  });
});
