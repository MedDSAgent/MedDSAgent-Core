import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// We test the CLI by running the compiled entry point or via tsx.
// Since the package is ESM, use `node --experimental-vm-modules` or tsx.
// Simplest: invoke `node dist/cli/index.js` after a build — but we don't
// want to require a build in tests. Use the source via `tsx` shim instead.

const CLI_ARGS_BASE = [
  "--input-type=module",
  "--import",
  "tsx/esm",                // tsx ESM hook — available via npx if not installed
];

// Detect tsx availability
function hasTsx(): boolean {
  const r = spawnSync("npx", ["tsx", "--version"], { encoding: "utf8", timeout: 5000 });
  return r.status === 0;
}

const TSX = hasTsx();

function runCli(args: string[], cwd?: string): { stdout: string; stderr: string; status: number | null } {
  // Run via: node --input-type=module --import tsx/esm src/cli/index.ts <args>
  const result = spawnSync(
    "npx",
    ["tsx", "src/cli/index.ts", ...args],
    {
      cwd: cwd ?? "/home/daviden1013/David_projects/MedDSAgent-Core",
      encoding: "utf8",
      timeout: 15000,
      env: { ...process.env, FORCE_COLOR: "0" },
    },
  );
  return {
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    status: result.status,
  };
}

describe.skipIf(!TSX)("CLI smoke tests", () => {
  let tmpDir: string;

  beforeAll(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "medds-cli-test-"));
  });

  afterAll(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it("--version prints version", () => {
    const { stdout, status } = runCli(["--version"]);
    expect(status).toBe(0);
    expect(stdout.trim()).toMatch(/\d+\.\d+\.\d+/);
  });

  it("--help lists subcommands", () => {
    const { stdout, status } = runCli(["--help"]);
    expect(status).toBe(0);
    expect(stdout).toContain("serve");
    expect(stdout).toContain("chat");
    expect(stdout).toContain("session");
  });

  it("serve --help shows options", () => {
    const { stdout, status } = runCli(["serve", "--help"]);
    expect(status).toBe(0);
    expect(stdout).toContain("--port");
    expect(stdout).toContain("--work-dir");
  });

  it("chat --help shows options", () => {
    const { stdout, status } = runCli(["chat", "--help"]);
    expect(status).toBe(0);
    expect(stdout).toContain("--model");
    expect(stdout).toContain("--language");
    expect(stdout).toContain("--query");
    expect(stdout).toContain("--no-resume");
  });

  it("session --help lists session subcommands", () => {
    const { stdout, status } = runCli(["session", "--help"]);
    expect(status).toBe(0);
    expect(stdout).toContain("list");
    expect(stdout).toContain("new");
    expect(stdout).toContain("delete");
  });

  it("session list shows 'No sessions found' on empty workspace", () => {
    const { stdout, status } = runCli(["session", "list", "--work-dir", tmpDir]);
    expect(status).toBe(0);
    expect(stdout).toContain("No sessions found");
  });
});
