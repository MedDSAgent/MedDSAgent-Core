import { describe, it, expect } from "vitest";
import { VERSION } from "../src/index.js";

describe("scaffold", () => {
  it("package exports VERSION", () => {
    expect(VERSION).toBe("0.1.0");
  });
});
