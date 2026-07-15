import * as fs from "fs";
import * as path from "path";
import type { ChatCompletionTool } from "openai/resources/chat/completions";
import { Tool } from "./index.js";

const ALLOWED_SUBDIRS = ["scripts", "uploads", "outputs", "internal"] as const;
const DEFAULT_MAX_READ = 1_048_576; // 1 MB

// ---------------------------------------------------------------------------
// FileSystemTool — restricted file system access (in-process, no subprocess)
// ---------------------------------------------------------------------------

export class FileSystemTool extends Tool {
  private readonly dir: string;
  private readonly maxReadSize: number;

  constructor(workDir: string, maxReadSize = DEFAULT_MAX_READ) {
    const description =
      "Interact with the file system. You can list, read, write, and delete files " +
      "within the 'scripts/', 'uploads/', 'outputs/', and 'internal/' directories. " +
      "Reads are text-only: use it for code, scripts, and plain-text data files. " +
      "Binary documents (PDF, DOCX, PPTX, XLSX) cannot be read this way — load those " +
      "with an appropriate library via the code execution tool instead.";
    super("FileSystem", description);
    this.dir = path.resolve(workDir);
    this.maxReadSize = maxReadSize;
  }

  override execute(params: Record<string, unknown>): string {
    const action = typeof params["action"] === "string" ? params["action"] : "list";
    const rawPath = typeof params["path"] === "string" ? params["path"] : "";

    try {
      switch (action) {
        case "list":
          return this._list(rawPath);
        case "read":
          return this._read(rawPath);
        case "write":
          return this._write(rawPath, typeof params["content"] === "string" ? params["content"] : "");
        case "delete":
          return this._delete(rawPath);
        default:
          return `Error: Unknown action '${action}'`;
      }
    } catch (err: unknown) {
      return `Error: ${err instanceof Error ? err.message : String(err)}`;
    }
  }

  override getTitle(args: Record<string, unknown>): string {
    const action = typeof args["action"] === "string" ? args["action"] : "";
    const p = typeof args["path"] === "string" ? args["path"] : "";
    switch (action) {
      case "list":   return p ? `List ${p}` : "List workspace";
      case "read":   return p ? `Read file ${p}` : "Read file";
      case "write":  return p ? `Write to ${p}` : "Write file";
      case "delete": return p ? `Delete ${p}` : "Delete file";
      default:       return action;
    }
  }

  override getToolCallSchema(): ChatCompletionTool {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.description,
        parameters: {
          type: "object",
          properties: {
            action: {
              type: "string",
              enum: ["list", "read", "write", "delete"],
              description:
                "Action to perform: list directory contents, read file content, " +
                "write content to a file, or delete a file/directory.",
            },
            path: {
              type: "string",
              description: "Relative path (e.g., 'uploads/data.csv' or 'outputs/report.md').",
            },
            content: {
              type: "string",
              description: "String content to write (required for 'write' action).",
            },
          },
          required: ["action"],
        },
      },
    };
  }

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------

  private _list(rawPath: string): string {
    if (!rawPath || rawPath.trim() === "" || rawPath === "." || rawPath === "/") {
      const entries: string[] = [];
      for (const subdir of ALLOWED_SUBDIRS) {
        const full = path.join(this.dir, subdir);
        if (fs.existsSync(full)) {
          const count = fs.readdirSync(full).length;
          entries.push(`[DIR]  ${subdir}/ (${count} items)`);
        }
      }
      return "Contents of Workspace:\n" + entries.join("\n");
    }

    const target = this._validatePath(rawPath);

    if (!fs.existsSync(target)) return `Path does not exist: ${rawPath}`;

    const stat = fs.statSync(target);
    if (!stat.isDirectory()) {
      return `File: ${path.basename(target)} (${this._formatSize(stat.size)})`;
    }

    const entries: string[] = [];
    for (const entry of fs.readdirSync(target).sort()) {
      if (entry.startsWith(".")) continue;
      const full = path.join(target, entry);
      const s = fs.statSync(full);
      if (s.isDirectory()) {
        entries.push(`[DIR]  ${entry}/ (${fs.readdirSync(full).length} items)`);
      } else {
        entries.push(`[FILE] ${entry} (${this._formatSize(s.size)})`);
      }
    }
    const rel = path.relative(this.dir, target);
    return `Contents of ${rel}:\n` + entries.join("\n");
  }

  private _read(rawPath: string): string {
    if (!rawPath) return "Error: 'path' is required for read action.";
    const target = this._validatePath(rawPath);
    if (!fs.existsSync(target)) return `Error: File does not exist: ${rawPath}`;
    if (fs.statSync(target).isDirectory()) return `Error: '${rawPath}' is a directory. Use 'list' instead.`;

    const size = fs.statSync(target).size;
    if (size > this.maxReadSize) {
      return (
        `Error: File '${rawPath}' is ${this._formatSize(size)}, ` +
        `which exceeds the ${this._formatSize(this.maxReadSize)} limit. ` +
        "For data files, use PythonExecutor to process in chunks."
      );
    }
    return fs.readFileSync(target, "utf8");
  }

  private _write(rawPath: string, content: string): string {
    if (!rawPath) return "Error: 'path' is required for write action.";
    const target = this._validatePath(rawPath);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, content, "utf8");
    return `Successfully wrote ${content.length} characters to ${rawPath}`;
  }

  private _delete(rawPath: string): string {
    if (!rawPath) return "Error: 'path' is required for delete action.";
    const target = this._validatePath(rawPath);
    if (!fs.existsSync(target)) return `Error: Path does not exist: ${rawPath}`;
    if (fs.statSync(target).isDirectory()) {
      fs.rmSync(target, { recursive: true });
      return `Successfully deleted directory: ${rawPath}`;
    }
    fs.unlinkSync(target);
    return `Successfully deleted file: ${rawPath}`;
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  private _validatePath(relativePath: string): string {
    const clean = path.normalize(relativePath);
    if (clean === "." || clean === "") return this.dir;

    const isAllowed = ALLOWED_SUBDIRS.some(
      (d) => clean === d || clean.startsWith(d + path.sep),
    );
    if (!isAllowed) {
      throw new Error(
        `Access Denied: restricted to [${ALLOWED_SUBDIRS.join(", ")}]. ` +
          `Cannot access '${clean}'.`,
      );
    }

    const target = path.resolve(this.dir, clean);
    if (target !== this.dir && !target.startsWith(this.dir + path.sep)) {
      throw new Error("Access denied: Cannot traverse outside workspace.");
    }
    return target;
  }

  private _formatSize(size: number): string {
    const units = ["B", "KB", "MB", "GB"] as const;
    let s = size;
    for (const unit of units) {
      if (s < 1024) return unit === "B" ? `${s} B` : `${s.toFixed(1)} ${unit}`;
      s /= 1024;
    }
    return `${s.toFixed(1)} TB`;
  }
}
