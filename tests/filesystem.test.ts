import { describe, it, expect, beforeEach, afterEach } from "vitest";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { FileSystemTool } from "../src/tools/filesystem.js";

let workDir: string;
let tool: FileSystemTool;

beforeEach(() => {
  workDir = fs.mkdtempSync(path.join(os.tmpdir(), "medds-fs-test-"));
  // Create allowed subdirectories
  for (const sub of ["scripts", "uploads", "outputs", "internal"]) {
    fs.mkdirSync(path.join(workDir, sub));
  }
  tool = new FileSystemTool(workDir);
});

afterEach(() => {
  fs.rmSync(workDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

describe("FileSystemTool list", () => {
  it("lists workspace root with all 4 subdirs", () => {
    const result = tool.execute({ action: "list" });
    expect(result).toContain("scripts/");
    expect(result).toContain("uploads/");
    expect(result).toContain("outputs/");
    expect(result).toContain("internal/");
  });

  it("lists contents of a subdir", () => {
    fs.writeFileSync(path.join(workDir, "scripts", "hello.py"), "print('hi')");
    const result = tool.execute({ action: "list", path: "scripts" });
    expect(result).toContain("hello.py");
  });

  it("returns info for a single file path", () => {
    fs.writeFileSync(path.join(workDir, "outputs", "report.txt"), "hello");
    const result = tool.execute({ action: "list", path: "outputs/report.txt" });
    expect(result).toContain("report.txt");
  });

  it("returns 'does not exist' for missing path", () => {
    const result = tool.execute({ action: "list", path: "scripts/missing" });
    expect(result).toContain("does not exist");
  });
});

// ---------------------------------------------------------------------------
// read
// ---------------------------------------------------------------------------

describe("FileSystemTool read", () => {
  it("reads a file's content", () => {
    fs.writeFileSync(path.join(workDir, "scripts", "code.py"), "x = 1");
    const result = tool.execute({ action: "read", path: "scripts/code.py" });
    expect(result).toBe("x = 1");
  });

  it("returns error for missing file", () => {
    const result = tool.execute({ action: "read", path: "scripts/nope.py" });
    expect(result).toContain("Error:");
    expect(result).toContain("does not exist");
  });

  it("returns error when reading a directory", () => {
    const result = tool.execute({ action: "read", path: "scripts" });
    expect(result).toContain("Error:");
    expect(result).toContain("directory");
  });

  it("returns error when file exceeds max read size", () => {
    const tinyTool = new FileSystemTool(workDir, 5); // 5 byte limit
    fs.writeFileSync(path.join(workDir, "scripts", "big.txt"), "hello world");
    const result = tinyTool.execute({ action: "read", path: "scripts/big.txt" });
    expect(result).toContain("Error:");
    expect(result).toContain("exceeds");
  });

  it("returns error when path parameter is missing", () => {
    expect(tool.execute({ action: "read" })).toContain("Error:");
  });
});

// ---------------------------------------------------------------------------
// write
// ---------------------------------------------------------------------------

describe("FileSystemTool write", () => {
  it("writes content and confirms character count", () => {
    const result = tool.execute({ action: "write", path: "outputs/out.txt", content: "hello" });
    expect(result).toContain("Successfully wrote");
    expect(fs.readFileSync(path.join(workDir, "outputs", "out.txt"), "utf8")).toBe("hello");
  });

  it("creates intermediate directories", () => {
    const result = tool.execute({ action: "write", path: "scripts/sub/file.py", content: "x=1" });
    expect(result).toContain("Successfully wrote");
    expect(fs.existsSync(path.join(workDir, "scripts", "sub", "file.py"))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// delete
// ---------------------------------------------------------------------------

describe("FileSystemTool delete", () => {
  it("deletes a file", () => {
    fs.writeFileSync(path.join(workDir, "outputs", "del.txt"), "bye");
    const result = tool.execute({ action: "delete", path: "outputs/del.txt" });
    expect(result).toContain("Successfully deleted");
    expect(fs.existsSync(path.join(workDir, "outputs", "del.txt"))).toBe(false);
  });

  it("deletes a directory recursively", () => {
    fs.mkdirSync(path.join(workDir, "scripts", "subdir"));
    fs.writeFileSync(path.join(workDir, "scripts", "subdir", "f.py"), "x");
    const result = tool.execute({ action: "delete", path: "scripts/subdir" });
    expect(result).toContain("Successfully deleted directory");
    expect(fs.existsSync(path.join(workDir, "scripts", "subdir"))).toBe(false);
  });

  it("returns error for missing path", () => {
    const result = tool.execute({ action: "delete", path: "outputs/nope.txt" });
    expect(result).toContain("Error:");
  });
});

// ---------------------------------------------------------------------------
// Security: path traversal
// ---------------------------------------------------------------------------

describe("FileSystemTool path traversal prevention", () => {
  it("blocks access to paths outside allowed subdirs", () => {
    const result = tool.execute({ action: "read", path: "../../etc/passwd" });
    expect(result).toContain("Error:");
    expect(result).toContain("Access");
  });

  it("blocks direct access to an arbitrary directory name", () => {
    const result = tool.execute({ action: "list", path: "secrets" });
    expect(result).toContain("Error:");
    expect(result).toContain("Access");
  });
});

// ---------------------------------------------------------------------------
// Schema and title
// ---------------------------------------------------------------------------

describe("FileSystemTool schema", () => {
  it("has correct name and required action param", () => {
    expect(tool.name).toBe("FileSystem");
    const schema = tool.getToolCallSchema();
    expect(schema.function.parameters?.required).toContain("action");
  });

  it("getTitle returns descriptive labels", () => {
    expect(tool.getTitle({ action: "list" })).toBe("List workspace");
    expect(tool.getTitle({ action: "list", path: "scripts" })).toBe("List scripts");
    expect(tool.getTitle({ action: "read", path: "uploads/data.csv" })).toContain("data.csv");
    expect(tool.getTitle({ action: "write", path: "outputs/r.md" })).toContain("r.md");
    expect(tool.getTitle({ action: "delete", path: "scripts/old.py" })).toContain("old.py");
  });
});
