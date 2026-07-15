import { describe, it, expect } from "vitest";
import { loadPrompt, applyTemplate } from "../src/util/prompts.js";

describe("applyTemplate", () => {
  it("replaces a single placeholder with a string value", () => {
    const tpl = "Hello, {{name}}!";
    expect(applyTemplate(tpl, "World")).toBe("Hello, World!");
  });

  it("replaces named placeholders with a dict", () => {
    const tpl = "{{greeting}}, {{name}}!";
    expect(applyTemplate(tpl, { greeting: "Hi", name: "Alice" })).toBe("Hi, Alice!");
  });

  it("throws when string value is given but template has 0 placeholders", () => {
    expect(() => applyTemplate("no placeholders", "value")).toThrow();
  });

  it("throws when string value is given but template has 2 placeholders", () => {
    expect(() => applyTemplate("{{a}} and {{b}}", "value")).toThrow();
  });

  it("throws when dict key is missing from template", () => {
    expect(() => applyTemplate("{{name}}", { wrong: "x" })).toThrow();
  });

  it("handles backslashes in replacement values without doubling", () => {
    const tpl = "Path: {{path}}";
    expect(applyTemplate(tpl, { path: "C:\\Users\\foo" })).toBe("Path: C:\\Users\\foo");
  });
});

describe("loadPrompt", () => {
  it("loads System_prompt_template and it is non-empty", () => {
    const content = loadPrompt("System_prompt_template");
    expect(content.length).toBeGreaterThan(100);
    expect(content).toContain("system");
  });

  it("loads a specialty prompt", () => {
    const content = loadPrompt("specialty/Biostatistics_Medical_Outcomes_Research");
    expect(content.length).toBeGreaterThan(50);
  });

  it("throws on missing prompt", () => {
    expect(() => loadPrompt("nonexistent_prompt")).toThrow();
  });
});
