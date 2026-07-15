import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { rmSync, mkdtempSync } from "fs";
import { InternalDatabase } from "../src/db/index.js";
import { makeUserStep, makeAgentStep, makeObservationStep } from "../src/history/index.js";

let tmpDir: string;
let db: InternalDatabase;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "meddsagent-test-"));
  db = new InternalDatabase(join(tmpDir, "test.db"));
});

afterEach(() => {
  db.close();
  rmSync(tmpDir, { recursive: true, force: true });
});

describe("Session CRUD", () => {
  it("creates and retrieves a session", () => {
    db.createSession("s1", "My Session", "/work/s1");
    const row = db.getSession("s1");
    expect(row?.session_id).toBe("s1");
    expect(row?.name).toBe("My Session");
    expect(row?.work_dir).toBe("/work/s1");
  });

  it("lists sessions in newest-first order", () => {
    db.createSession("a", "A", "/a");
    db.createSession("b", "B", "/b");
    const list = db.listSessions();
    expect(list[0]?.session_id).toBe("b");
    expect(list[1]?.session_id).toBe("a");
  });

  it("renames a session", () => {
    db.createSession("s1", "Old", "/w");
    db.renameSession("s1", "New");
    expect(db.getSession("s1")?.name).toBe("New");
  });

  it("deletes a session and cascades history", () => {
    db.createSession("s1", "X", "/x");
    db.addHistoryStep("s1", 1, makeUserStep("hi"));
    db.deleteSession("s1");
    expect(db.getSession("s1")).toBeUndefined();
    expect(db.getHistorySteps("s1")).toHaveLength(0);
  });

  it("saves and retrieves config", () => {
    db.createSession("s1", "X", "/x");
    db.saveSessionConfig("s1", { llm: "gpt-4o", temp: 0.7 });
    const cfg = db.getSessionConfig("s1");
    expect(cfg?.["llm"]).toBe("gpt-4o");
  });

  it("merges specialty into config", () => {
    db.createSession("s1", "X", "/x");
    db.saveSessionConfig("s1", { llm: "gpt-4o" });
    db.saveSessionSpecialty("s1", "biostat", "biostat instructions");
    const cfg = db.getSessionConfig("s1");
    expect(cfg?.["specialty_id"]).toBe("biostat");
    expect(cfg?.["specialty_prompt"]).toBe("biostat instructions");
  });
});

describe("History CRUD", () => {
  beforeEach(() => db.createSession("s1", "Test", "/w"));

  it("stores and retrieves UserStep", () => {
    const step = makeUserStep("what is 1+1?");
    db.addHistoryStep("s1", 1, step);
    const steps = db.getHistorySteps("s1");
    expect(steps).toHaveLength(1);
    expect(steps[0]?.type).toBe("UserStep");
  });

  it("preserves step order", () => {
    db.addHistoryStep("s1", 1, makeUserStep("q"));
    db.addHistoryStep(
      "s1",
      1,
      makeAgentStep("a1", {
        tools: [{ tool_name: "PythonExecutor", tool_args: '{}', tool_title: "" }],
      }),
    );
    db.addHistoryStep("s1", 1, makeObservationStep("a1", [{ tool_name: "PythonExecutor", output: "2" }]));
    const steps = db.getHistorySteps("s1");
    expect(steps.map((s) => s.type)).toEqual(["UserStep", "AgentStep", "ObservationStep"]);
  });

  it("round-trips AgentStep isFinal flag", () => {
    const step = makeAgentStep("a1", { isFinal: true, tools: [{ tool_name: "final_response", tool_args: "{}", tool_title: "" }] });
    db.addHistoryStep("s1", 1, step);
    const [retrieved] = db.getHistorySteps("s1");
    expect(retrieved?.type).toBe("AgentStep");
    if (retrieved?.type === "AgentStep") {
      expect(retrieved.isFinal).toBe(true);
    }
  });
});

describe("Session State", () => {
  beforeEach(() => db.createSession("s1", "Test", "/w"));

  it("upserts session state", () => {
    db.saveSessionState("s1", { memory: "full" }, "/path/state.pkl");
    const s = db.getSessionState("s1");
    expect(s?.python_state_path).toBe("/path/state.pkl");
  });

  it("overwrites on second save", () => {
    db.saveSessionState("s1", { memory: "full" }, "/old.pkl");
    db.saveSessionState("s1", { memory: "sliding" }, "/new.pkl");
    expect(db.getSessionState("s1")?.python_state_path).toBe("/new.pkl");
  });

  it("returns null for missing state", () => {
    expect(db.getSessionState("s1")).toBeNull();
  });
});
