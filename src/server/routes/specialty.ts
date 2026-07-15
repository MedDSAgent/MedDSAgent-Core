import * as fs from "fs";
import * as path from "path";
import type { FastifyPluginAsync } from "fastify";

// Resolve the prompts/specialty directory, probing dist/ first then source.
function resolveSpecialtyDir(): string {
  const candidates = [
    path.resolve(import.meta.dirname, "../../..", "dist", "prompts", "specialty"),
    path.resolve(import.meta.dirname, "../../..", "prompts", "specialty"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return candidates[candidates.length - 1]!;
}

let _specialtyDir: string | undefined;
function getSpecialtyDir(): string {
  return (_specialtyDir ??= resolveSpecialtyDir());
}

interface SpecialtyEntry {
  id: string;
  display_name: string;
  filename: string;
}

function loadIndex(): SpecialtyEntry[] {
  const indexPath = path.join(getSpecialtyDir(), "index.json");
  if (!fs.existsSync(indexPath)) return [];
  return JSON.parse(fs.readFileSync(indexPath, "utf8")) as SpecialtyEntry[];
}

export const specialtyRoutes: FastifyPluginAsync = async (fastify) => {
  fastify.get("/specialty-prompts", async () => {
    return loadIndex();
  });

  fastify.get<{ Params: { prompt_id: string } }>(
    "/specialty-prompts/:prompt_id",
    async (request, reply) => {
      const { prompt_id } = request.params;
      const index = loadIndex();
      const entry = index.find((e) => e.id === prompt_id);
      if (!entry) {
        reply.code(404).send({ detail: "Specialty prompt not found" });
        return;
      }
      const filePath = path.join(getSpecialtyDir(), entry.filename);
      if (!fs.existsSync(filePath)) {
        reply.code(404).send({ detail: "Specialty prompt file not found" });
        return;
      }
      return {
        id: entry.id,
        display_name: entry.display_name,
        content: fs.readFileSync(filePath, "utf8"),
      };
    },
  );
};
