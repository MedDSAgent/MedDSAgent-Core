#!/usr/bin/env node
import { Command } from "commander";
import * as readline from "readline";
import * as fs from "fs";
import * as path from "path";
import { SessionManager } from "../session/index.js";
import { startServer as _startServer } from "../server/index.js";
import { VERSION } from "../index.js";
import type { SessionConfig } from "../server/types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function env(key: string, fallback = ""): string {
  return process.env[key] ?? fallback;
}

function resolveWorkDir(raw: string): string {
  return path.resolve(raw);
}

/** Ask a single readline question, returns the user's answer. */
function ask(rl: readline.Interface, prompt: string): Promise<string> {
  return new Promise((resolve) => rl.question(prompt, resolve));
}

// ---------------------------------------------------------------------------
// REPL
// ---------------------------------------------------------------------------

async function runRepl(manager: SessionManager, sessionId: string): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: process.stdin.isTTY,
  });

  const printBanner = () => {
    console.log("\nMedDSAgent  —  type 'exit' to quit, 'help' for commands\n");
  };

  const printHelp = () => {
    console.log("  exit     Quit");
    console.log("  reset    Clear history and restart worker state");
    console.log("  state    Show current variable state");
    console.log("  history  Show conversation history summary");
    console.log("  help     Show this message\n");
  };

  if (process.stdin.isTTY) printBanner();

  // Ctrl-C inside a running chat should stop it, not kill the process.
  let stopCurrentChat: (() => void) | null = null;
  process.on("SIGINT", () => {
    if (stopCurrentChat) {
      stopCurrentChat();
    } else {
      rl.close();
    }
  });

  try {
    while (true) {
      let input: string;
      try {
        input = await ask(rl, process.stdin.isTTY ? "\x1b[1;32mYou\x1b[0m: " : "");
      } catch {
        break; // EOF
      }

      const cmd = input.trim().toLowerCase();
      if (cmd === "" ) continue;
      if (cmd === "exit" || cmd === "quit") {
        if (process.stdin.isTTY) console.log("Goodbye!");
        break;
      }
      if (cmd === "help") { printHelp(); continue; }

      if (cmd === "state") {
        const vars = await manager.getVariables(sessionId);
        if (!vars || (!vars.python?.length && !vars.r?.length)) {
          console.log("  (no variables defined)");
        } else {
          const list = (vars.python ?? vars.r ?? []) as Array<Record<string, unknown>>;
          for (const v of list) {
            console.log(`  ${String(v["name"])}  (${String(v["type"])})  ${String(v["value"] ?? "")}`);
          }
        }
        console.log();
        continue;
      }

      if (cmd === "history") {
        const steps = manager.getHistory(sessionId);
        const rounds = new Set(steps.map((s) => ("roundNum" in s ? s.roundNum : 0)));
        console.log(`  ${rounds.size} round(s), ${steps.length} total step(s)\n`);
        continue;
      }

      if (cmd === "reset") {
        await manager.deleteSession(sessionId);
        const detail = await manager.getSession(sessionId);
        const config = detail?.config ?? {};
        sessionId = await manager.createSession("CLI Session", config);
        console.log("  History and state cleared.\n");
        continue;
      }

      // --- Chat ---
      let aborted = false;
      const ctl = new AbortController();
      stopCurrentChat = () => {
        aborted = true;
        ctl.abort();
        void manager.stopSession(sessionId);
      };

      let agentStarted = false;
      try {
        for await (const event of manager.chat(sessionId, input.trim())) {
          if (ctl.signal.aborted) break;

          switch (event.type) {
            case "response":
              if (!agentStarted) {
                if (process.stdin.isTTY) process.stdout.write("\x1b[1;34mAgent\x1b[0m: ");
                agentStarted = true;
              }
              process.stdout.write(event.data);
              break;
            case "tool_running":
              if (process.stdin.isTTY) {
                console.log(`\n\x1b[33m[${event.toolName}]\x1b[0m ${event.data["title"] ?? ""}`);
              }
              break;
            case "tool_output":
              if (process.stdin.isTTY) {
                const preview = event.data.slice(0, 500);
                console.log(preview + (event.data.length > 500 ? "\n  …(truncated)" : ""));
              }
              break;
            case "final_decision":
              if (event.isFinal && agentStarted) {
                process.stdout.write("\n");
              }
              break;
          }
        }
      } catch (err: unknown) {
        if (!aborted) {
          console.error(`\nError: ${err instanceof Error ? err.message : String(err)}`);
        }
      } finally {
        stopCurrentChat = null;
        if (agentStarted && !aborted) console.log();
      }
    }
  } finally {
    rl.close();
  }
}

// ---------------------------------------------------------------------------
// Config builder from CLI flags
// ---------------------------------------------------------------------------

function buildConfig(opts: Record<string, string | undefined>): SessionConfig {
  const cfg: SessionConfig = {};
  if (opts["provider"]) cfg.llm_provider = opts["provider"];
  if (opts["model"]) cfg.llm_model = opts["model"];
  if (opts["apiKey"]) cfg.llm_api_key = opts["apiKey"];
  if (opts["baseUrl"]) cfg.llm_base_url = opts["baseUrl"];
  if (opts["apiVersion"]) cfg.llm_api_version = opts["apiVersion"];
  if (opts["language"]) cfg.language = opts["language"];
  if (opts["pythonBin"]) cfg.python_bin = opts["pythonBin"];
  if (opts["rHome"]) cfg.r_home = opts["rHome"];
  if (opts["reasoningEffort"]) cfg.reasoning_effort = opts["reasoningEffort"];
  if (opts["temperature"]) cfg.temperature = parseFloat(opts["temperature"]);
  if (opts["topP"]) cfg.top_p = parseFloat(opts["topP"]);
  if (opts["specialtyPrompt"]) cfg.specialty_prompt = opts["specialtyPrompt"];
  if (opts["dbConnectionCode"]) cfg.db_connection_code = opts["dbConnectionCode"];
  return cfg;
}

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

const program = new Command();

program
  .name("meddsagent")
  .description("MedDSAgent — Medical Data Science Agent")
  .version(VERSION);

// ---- serve ----------------------------------------------------------------

program
  .command("serve")
  .description("Start the Fastify HTTP server")
  .option("-w, --work-dir <dir>",
    "Workspace directory (sessions stored under <dir>/sessions/)",
    env("WORK_DIR", "./workspace"))
  .option("-p, --port <number>", "HTTP port", env("PORT", "7842"))
  .option("--host <host>",       "HTTP host", env("HOST", "0.0.0.0"))
  .action(async (opts: { workDir: string; port: string; host: string }) => {
    const manager = new SessionManager(resolveWorkDir(opts.workDir));
    const handle = await _startServer(manager, parseInt(opts.port, 10), opts.host);
    console.log(`Server running at ${handle.address}`);

    const shutdown = () => {
      handle.close().then(() => process.exit(0)).catch(() => process.exit(1));
    };
    process.on("SIGTERM", shutdown);
    process.on("SIGINT", shutdown);
  });

// ---- chat -----------------------------------------------------------------

program
  .command("chat")
  .description("Start an interactive REPL chat session")
  .option("-w, --work-dir <dir>", "Workspace directory", env("WORK_DIR", "./workspace"))
  .option("--provider <p>",         "LLM provider: openai|azure",         env("LLM_PROVIDER", "openai"))
  .option("-m, --model <m>",        "LLM model name",                     env("LLM_MODEL", "gpt-4o"))
  .option("--api-key <k>",          "LLM API key",                        env("LLM_API_KEY", env("OPENAI_API_KEY")))
  .option("--base-url <u>",         "LLM base URL / Azure endpoint",      env("LLM_BASE_URL"))
  .option("--api-version <v>",      "Azure API version",                  env("LLM_API_VERSION"))
  .option("--language <lang>",      "Executor language: python|r",        env("LANGUAGE", "python"))
  .option("--python-bin <p>",       "Python executable path",             env("MEDDS_PYTHON_BIN"))
  .option("--r-home <r>",           "R_HOME for R sessions",              env("MEDDS_R_HOME"))
  .option("--reasoning-effort <e>", "Reasoning effort: low|medium|high",  env("LLM_REASONING_EFFORT"))
  .option("--temperature <t>",      "Sampling temperature",               env("LLM_TEMPERATURE", "1.0"))
  .option("--specialty-prompt <p>", "Domain-specific system prompt addition")
  .option("--db-connection-code <c>", "Code to establish a DB connection", env("DB_CONNECTION_CODE"))
  .option("--db-connection-file <f>", "File containing DB connection code", env("DB_CONNECTION_FILE"))
  .option("--no-resume",            "Start a fresh session (ignore existing)")
  .option("-q, --query <q>",        "Run a single query and exit (non-interactive)")
  .action(async (opts: Record<string, string | boolean | undefined>) => {
    const workDir = resolveWorkDir((opts["workDir"] as string | undefined) ?? "./workspace");

    // Resolve DB connection code
    let dbCode = opts["dbConnectionCode"] as string | undefined;
    const dbFile = opts["dbConnectionFile"] as string | undefined;
    if (dbFile) {
      try {
        dbCode = fs.readFileSync(dbFile, "utf8");
      } catch (err) {
        console.error(`Failed to read DB connection file: ${err instanceof Error ? err.message : String(err)}`);
        process.exit(1);
      }
    }
    if (dbCode) opts["dbConnectionCode"] = dbCode;

    const config = buildConfig(opts as Record<string, string | undefined>);
    const manager = new SessionManager(workDir);

    // Find or create session
    let sessionId: string;
    const resume = opts["resume"] !== false; // commander sets --no-resume → resume: false
    if (resume) {
      const sessions = await manager.listSessions();
      if (sessions.length > 0 && sessions[0]) {
        sessionId = sessions[0].session_id;
        if (process.stdin.isTTY) {
          console.log(`Resuming session: ${sessions[0].name} (${sessionId.slice(0, 8)}…)`);
        }
      } else {
        sessionId = await manager.createSession("CLI Session", config);
      }
    } else {
      sessionId = await manager.createSession("CLI Session", config);
    }

    const query = opts["query"] as string | undefined;
    if (query) {
      // Single-query mode
      for await (const event of manager.chat(sessionId, query)) {
        if (event.type === "response") process.stdout.write(event.data);
        if (event.type === "tool_output") console.log(event.data);
      }
      console.log();
    } else {
      await runRepl(manager, sessionId);
    }
  });

// ---- session --------------------------------------------------------------

const sessionCmd = program.command("session").description("Manage sessions");

sessionCmd
  .command("list")
  .description("List all sessions in a workspace")
  .option("-w, --work-dir <dir>", "Workspace directory", env("WORK_DIR", "./workspace"))
  .action(async (opts: { workDir: string }) => {
    const manager = new SessionManager(resolveWorkDir(opts.workDir));
    const sessions = await manager.listSessions();
    if (sessions.length === 0) {
      console.log("No sessions found.");
    } else {
      const idW = 36, nameW = 28;
      console.log(`\n${"SESSION ID".padEnd(idW)}  ${"NAME".padEnd(nameW)}  LAST ACCESSED`);
      console.log("─".repeat(idW + nameW + 22));
      for (const s of sessions) {
        const accessed = new Date(s.last_accessed).toLocaleString();
        console.log(`${s.session_id.padEnd(idW)}  ${s.name.slice(0, nameW).padEnd(nameW)}  ${accessed}`);
      }
      console.log();
    }
  });

sessionCmd
  .command("new")
  .description("Create a new session and open a REPL")
  .option("-w, --work-dir <dir>",   "Workspace directory",    env("WORK_DIR", "./workspace"))
  .option("-n, --name <name>",      "Session name",           "CLI Session")
  .option("--provider <p>",         "LLM provider",           env("LLM_PROVIDER", "openai"))
  .option("-m, --model <m>",        "LLM model name",         env("LLM_MODEL", "gpt-4o"))
  .option("--api-key <k>",          "LLM API key",            env("LLM_API_KEY", env("OPENAI_API_KEY")))
  .option("--base-url <u>",         "LLM base URL",           env("LLM_BASE_URL"))
  .option("--api-version <v>",      "Azure API version",      env("LLM_API_VERSION"))
  .option("--language <lang>",      "Executor language",      env("LANGUAGE", "python"))
  .option("--python-bin <p>",       "Python executable path", env("MEDDS_PYTHON_BIN"))
  .option("--r-home <r>",           "R_HOME for R sessions",  env("MEDDS_R_HOME"))
  .option("--specialty-prompt <p>", "Specialty prompt text")
  .option("--db-connection-code <c>", "DB connection code",   env("DB_CONNECTION_CODE"))
  .option("--db-connection-file <f>", "File with DB connection code")
  .action(async (opts: Record<string, string | undefined>) => {
    const workDir = resolveWorkDir(opts["workDir"] ?? "./workspace");

    let dbCode = opts["dbConnectionCode"];
    if (opts["dbConnectionFile"]) {
      try {
        dbCode = fs.readFileSync(opts["dbConnectionFile"], "utf8");
      } catch (err) {
        console.error(`Failed to read DB connection file: ${err instanceof Error ? err.message : String(err)}`);
        process.exit(1);
      }
    }
    if (dbCode) opts["dbConnectionCode"] = dbCode;

    const manager = new SessionManager(workDir);
    const config = buildConfig(opts);
    const name = opts["name"] ?? "CLI Session";
    const sessionId = await manager.createSession(name, config);
    if (process.stdin.isTTY) console.log(`Created session: ${sessionId}\n`);
    await runRepl(manager, sessionId);
  });

sessionCmd
  .command("delete <id>")
  .description("Delete a session and remove its data")
  .option("-w, --work-dir <dir>", "Workspace directory", env("WORK_DIR", "./workspace"))
  .action(async (id: string, opts: { workDir: string }) => {
    const manager = new SessionManager(resolveWorkDir(opts.workDir));
    await manager.deleteSession(id);
    console.log(`Deleted session: ${id}`);
  });

program.parse();
