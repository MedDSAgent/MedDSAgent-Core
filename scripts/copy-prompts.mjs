// Copies prompts/ into dist/prompts/ and static assets into their dist locations.
import { cpSync, copyFileSync, mkdirSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

// Prompts
const promptsSrc = join(root, "prompts");
const promptsDst = join(root, "dist", "prompts");
mkdirSync(promptsDst, { recursive: true });
cpSync(promptsSrc, promptsDst, { recursive: true });
console.log("Copied prompts/ → dist/prompts/");

// DB schema (tsc doesn't copy non-TS files)
const schemaSrc = join(root, "src", "db", "schema.sql");
const schemaDst = join(root, "dist", "db", "schema.sql");
mkdirSync(join(root, "dist", "db"), { recursive: true });
copyFileSync(schemaSrc, schemaDst);
console.log("Copied src/db/schema.sql → dist/db/schema.sql");
