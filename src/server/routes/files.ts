import * as fs from "fs";
import * as path from "path";
import { spawn } from "child_process";
import { createReadStream } from "fs";
import type { FastifyPluginAsync } from "fastify";
import type { ISessionManager } from "../types.js";

// Text extensions that can be read/edited as UTF-8
const TEXT_EXTS = new Set([
  ".py", ".r", ".R", ".sql", ".txt", ".md", ".json", ".csv", ".tsv",
  ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash",
  ".js", ".ts", ".html", ".css", ".xml", ".log", ".env", ".gitignore", ".data",
]);
const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"]);
const MAX_PREVIEW_LINES = 1000;

function formatSize(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"] as const;
  let s = bytes;
  for (const u of units) {
    if (s < 1024) return u === "B" ? `${s} B` : `${s.toFixed(1)} ${u}`;
    s /= 1024;
  }
  return `${s.toFixed(1)} TB`;
}

interface Opts {
  manager: ISessionManager;
}

export const filesRoutes: FastifyPluginAsync<Opts> = async (fastify, opts) => {
  const { manager } = opts;

  // ------------------------------------------------------------------
  // Helper: resolve session-scoped path (null = traversal detected or not found)
  // ------------------------------------------------------------------

  function sessionPath(sessionId: string, subPath = ""): string | null {
    const sessionDir = path.resolve(manager.sessionsDir, sessionId);
    if (!fs.existsSync(sessionDir)) return null;
    if (!subPath) return sessionDir;
    const target = path.resolve(sessionDir, subPath);
    if (target !== sessionDir && !target.startsWith(sessionDir + path.sep)) return null;
    return target;
  }

  // ------------------------------------------------------------------
  // GET /sessions/:id/files — list files
  // ------------------------------------------------------------------

  fastify.get<{ Params: { session_id: string }; Querystring: { path?: string } }>(
    "/sessions/:session_id/files",
    async (request, reply) => {
      try {
        const { session_id } = request.params;
        const subPath = request.query.path ?? "";
        const target = sessionPath(session_id, subPath);
        if (target === null) { reply.code(404).send({ detail: "Session not found" }); return; }
        if (!fs.existsSync(target)) return [];

        const sessionDir = path.resolve(manager.sessionsDir, session_id);
        const files: unknown[] = [];
        for (const entry of fs.readdirSync(target).sort()) {
          if (entry.startsWith(".") || ["state.pkl", "state.RData", "internal"].includes(entry))
            continue;
          const full = path.join(target, entry);
          const stat = fs.statSync(full);
          files.push({
            name: entry,
            path: path.relative(sessionDir, full),
            size: stat.size,
            size_human: formatSize(stat.size),
            is_directory: stat.isDirectory(),
            modified_at: new Date(stat.mtimeMs).toISOString(),
          });
        }
        return files;
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // ------------------------------------------------------------------
  // POST /sessions/:id/files — upload
  // ------------------------------------------------------------------

  fastify.post<{ Params: { session_id: string }; Querystring: { path?: string } }>(
    "/sessions/:session_id/files",
    async (request, reply) => {
      const { session_id } = request.params;
      const subDir = request.query.path ?? "uploads";
      const targetDir = sessionPath(session_id, subDir);
      if (targetDir === null) { reply.code(404).send({ detail: "Session not found" }); return; }

      let finalPath: string | undefined;
      try {
        const part = await request.file();
        if (!part) { reply.code(400).send({ detail: "No file in request." }); return; }
        fs.mkdirSync(targetDir, { recursive: true });
        finalPath = path.join(targetDir, part.filename);
        const writeStream = fs.createWriteStream(finalPath);
        await new Promise<void>((resolve, reject) => {
          part.file.pipe(writeStream);
          writeStream.on("finish", resolve);
          writeStream.on("error", reject);
        });
        await manager.addSystemMessage(session_id, `File '${part.filename}' uploaded to '${subDir}'.`);
        return { status: "uploaded", filename: part.filename };
      } catch (err: unknown) {
        if (finalPath && fs.existsSync(finalPath)) fs.unlinkSync(finalPath);
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // ------------------------------------------------------------------
  // POST /sessions/:id/index — document indexing (deferred)
  // ------------------------------------------------------------------

  fastify.post("/sessions/:session_id/index", async (_request, reply) => {
    reply.code(501).send({ detail: "Document indexing is not yet available in the TS backend." });
  });

  // ------------------------------------------------------------------
  // GET /sessions/:id/files/:file_name/index-status
  // ------------------------------------------------------------------

  fastify.get<{ Params: { session_id: string; file_name: string } }>(
    "/sessions/:session_id/files/:file_name/index-status",
    async (request, reply) => {
      try {
        const { session_id, file_name } = request.params;
        const doc = manager.db.getParsedDocument(session_id, file_name);
        if (!doc) {
          return { file_name, status: "not_indexed", section_count: 0, error_message: null };
        }
        const sectionCount =
          doc.status === "done" ? manager.db.getDocumentSections(doc.document_id).length : 0;
        return {
          file_name,
          status: doc.status,
          section_count: sectionCount,
          error_message: doc.error_message ?? null,
        };
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // ------------------------------------------------------------------
  // GET /sessions/:id/files/content?path=<file_path> — read for in-browser editor
  // (registered BEFORE the catch-all download route)
  // ------------------------------------------------------------------

  fastify.get<{ Params: { session_id: string }; Querystring: { path?: string } }>(
    "/sessions/:session_id/files/content",
    async (request, reply) => {
      const { session_id } = request.params;
      const filePath = request.query.path ?? "";
      const target = sessionPath(session_id, filePath);
      if (target === null) { reply.code(403).send({ detail: "Access denied" }); return; }
      if (!fs.existsSync(target)) { reply.code(404).send({ detail: "File not found" }); return; }
      if (fs.statSync(target).isDirectory()) {
        reply.code(400).send({ detail: "Cannot read directory content" });
        return;
      }

      const ext = path.extname(filePath).toLowerCase();

      if (IMAGE_EXTS.has(ext)) {
        const b64 = fs.readFileSync(target).toString("base64");
        const mime = ext === ".svg" ? "image/svg+xml" : `image/${ext.slice(1)}`;
        return { content: `data:${mime};base64,${b64}`, file_type: "image", is_truncated: false, extension: ext };
      }

      if (TEXT_EXTS.has(ext) || ext === "") {
        const lines = fs.readFileSync(target, "utf8").split("\n");
        const truncated = lines.length > MAX_PREVIEW_LINES;
        return {
          content: (truncated ? lines.slice(0, MAX_PREVIEW_LINES) : lines).join("\n"),
          file_type: "text",
          is_truncated: truncated,
          total_lines: lines.length,
          extension: ext,
        };
      }

      return { content: "", file_type: "binary", is_truncated: false, extension: ext };
    },
  );

  // ------------------------------------------------------------------
  // PUT /sessions/:id/files/content?path=<file_path> — save for in-browser editor
  // ------------------------------------------------------------------

  fastify.put<{ Params: { session_id: string }; Querystring: { path?: string }; Body: { content: string } }>(
    "/sessions/:session_id/files/content",
    async (request, reply) => {
      const { session_id } = request.params;
      const filePath = request.query.path ?? "";
      const target = sessionPath(session_id, filePath);
      if (target === null) { reply.code(403).send({ detail: "Access denied" }); return; }
      if (!fs.existsSync(target)) { reply.code(404).send({ detail: "File not found" }); return; }
      if (fs.statSync(target).isDirectory()) {
        reply.code(400).send({ detail: "Cannot write to directory" });
        return;
      }
      const ext = path.extname(filePath).toLowerCase();
      if (!TEXT_EXTS.has(ext) && ext !== "") {
        reply.code(400).send({ detail: "Cannot edit this file type" });
        return;
      }
      try {
        fs.writeFileSync(target, request.body.content, "utf8");
        await manager.addSystemMessage(session_id, `User edited file '${filePath}'.`);
        return { status: "saved", file_path: filePath };
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // ------------------------------------------------------------------
  // DELETE /sessions/:id/files/* — delete file or directory
  // ------------------------------------------------------------------

  fastify.delete<{ Params: { session_id: string; "*": string } }>(
    "/sessions/:session_id/files/*",
    async (request, reply) => {
      const { session_id } = request.params;
      const filePath = request.params["*"] ?? "";
      const target = sessionPath(session_id, filePath);
      if (target === null) { reply.code(403).send({ detail: "Access denied" }); return; }
      if (!fs.existsSync(target)) { reply.code(404).send({ detail: "File not found" }); return; }
      try {
        const isDir = fs.statSync(target).isDirectory();
        if (isDir) fs.rmSync(target, { recursive: true });
        else fs.unlinkSync(target);
        await manager.addSystemMessage(
          session_id,
          `User deleted ${isDir ? "folder" : "file"} '${filePath}'.`,
        );
        return { status: "deleted" };
      } catch (err: unknown) {
        reply.code(500).send({ detail: err instanceof Error ? err.message : String(err) });
      }
    },
  );

  // ------------------------------------------------------------------
  // GET /sessions/:id/files/* — download (catch-all, must be last)
  // ------------------------------------------------------------------

  fastify.get<{ Params: { session_id: string; "*": string } }>(
    "/sessions/:session_id/files/*",
    async (request, reply) => {
      const { session_id } = request.params;
      const filePath = request.params["*"] ?? "";
      const target = sessionPath(session_id, filePath);
      if (target === null) { reply.code(403).send({ detail: "Access denied" }); return; }
      if (!fs.existsSync(target)) { reply.code(404).send({ detail: "File not found" }); return; }

      if (fs.statSync(target).isDirectory()) {
        // Stream as tar.gz via system tar command (Linux/macOS)
        const basename = path.basename(target);
        const parentDir = path.dirname(target);
        reply.raw.setHeader("Content-Type", "application/gzip");
        reply.raw.setHeader("Content-Disposition", `attachment; filename="${basename}.tar.gz"`);
        const tarProc = spawn("tar", ["-czf", "-", "-C", parentDir, basename], {
          stdio: ["ignore", "pipe", "ignore"],
        });
        tarProc.stdout.pipe(reply.raw);
        await new Promise<void>((resolve) => tarProc.once("close", resolve));
        return reply;
      }

      const filename = path.basename(target);
      reply.header("Content-Disposition", `attachment; filename="${filename}"`);
      reply.type("application/octet-stream");
      return reply.send(createReadStream(target));
    },
  );
};
