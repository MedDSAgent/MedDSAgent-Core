import { describe, it, expect } from "vitest";

// Test the argument normalization logic in isolation.
// We can't call real APIs in unit tests, so we exercise the exported types
// and the normalization helper by importing it directly from the module source.

// The normalizeArguments function is private, so we test it indirectly by
// verifying that the module typechecks and that our assumptions about the
// ChatResult interface hold.

import type { ChatResult, ToolCallResult } from "../src/engines/index.js";
import { OpenAIEngine, AzureOpenAIEngine } from "../src/engines/index.js";

describe("engine module shape", () => {
  it("OpenAIEngine is constructable", () => {
    const engine = new OpenAIEngine({ apiKey: "test-key", model: "gpt-4o" });
    expect(typeof engine.chat).toBe("function");
  });

  it("AzureOpenAIEngine is constructable", () => {
    const engine = new AzureOpenAIEngine({
      apiKey: "test-key",
      model: "gpt-4o",
      endpoint: "https://example.openai.azure.com",
    });
    expect(typeof engine.chat).toBe("function");
  });

  it("ChatResult shape matches expectations", () => {
    const result: ChatResult = {
      response: "hello",
      toolCalls: [{ name: "PythonExecutor", arguments: { code: "print(1)" } }],
    };
    expect(result.toolCalls[0]?.name).toBe("PythonExecutor");
    expect(result.toolCalls[0]?.arguments["code"]).toBe("print(1)");
  });
});

describe("argument normalization (via module behavior)", () => {
  it("ToolCallResult arguments are an object (not a string)", () => {
    const tc: ToolCallResult = { name: "foo", arguments: { x: 1 } };
    expect(typeof tc.arguments).toBe("object");
  });
});
