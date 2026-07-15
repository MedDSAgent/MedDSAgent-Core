import { readFileSync } from "fs";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";
import { existsSync } from "fs";

// ---------------------------------------------------------------------------
// Prompt directory resolution
//
// Probe locations in order:
//   1. dist/prompts/   (bundled — __dirname is dist/util/)
//   2. prompts/        (dev — __dirname is src/util/, go up two levels)
// ---------------------------------------------------------------------------

const __dirname = dirname(fileURLToPath(import.meta.url));

function findPromptsDir(): string {
  // When compiled: dist/util → dist/prompts
  const distPath = resolve(__dirname, "..", "prompts");
  if (existsSync(distPath)) return distPath;

  // Dev mode: src/util → repo root → prompts
  const devPath = resolve(__dirname, "..", "..", "prompts");
  if (existsSync(devPath)) return devPath;

  throw new Error(
    `Could not locate prompts/ directory. Searched:\n  ${distPath}\n  ${devPath}`,
  );
}

let _promptsDir: string | undefined;

function promptsDir(): string {
  return (_promptsDir ??= findPromptsDir());
}

// ---------------------------------------------------------------------------
// loadPrompt
//
// name examples:  "System_prompt_template"
//                 "specialty/Biostatistics_Medical_Outcomes_Research"
// ---------------------------------------------------------------------------

export function loadPrompt(name: string): string {
  const filePath = join(promptsDir(), `${name}.md`);
  try {
    return readFileSync(filePath, "utf-8");
  } catch (cause) {
    throw new Error(`Prompt not found: "${name}" (looked at ${filePath})`, { cause });
  }
}

// ---------------------------------------------------------------------------
// applyTemplate
//
// Replaces {{placeholder}} tokens in a template.
//
// applyTemplate(tpl, "text")           — single placeholder, any name
// applyTemplate(tpl, { key: "value" }) — named placeholders
// ---------------------------------------------------------------------------

export function applyTemplate(template: string, vars: string | Record<string, string>): string {
  const pattern = /\{\{(.*?)\}\}/g;

  if (typeof vars === "string") {
    const matches = [...template.matchAll(pattern)];
    if (matches.length !== 1) {
      throw new Error(
        `applyTemplate: string value requires exactly 1 placeholder, found ${matches.length}.`,
      );
    }
    return template.replace(pattern, () => vars);
  }

  const placeholders = [...template.matchAll(pattern)].map((m) => m[1] as string);
  const varKeys = Object.keys(vars);

  if (placeholders.length !== varKeys.length) {
    throw new Error(
      `applyTemplate: template has ${placeholders.length} placeholder(s) but vars has ${varKeys.length} key(s).`,
    );
  }
  for (const key of varKeys) {
    if (!placeholders.includes(key)) {
      throw new Error(`applyTemplate: key "${key}" not found in template placeholders.`);
    }
  }

  return template.replace(pattern, (_match, name: string) => {
    const val = vars[name];
    if (val === undefined) throw new Error(`applyTemplate: missing value for placeholder "{{${name}}}".`);
    return val;
  });
}
