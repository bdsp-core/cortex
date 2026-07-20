import { readFileSync, readdirSync } from "node:fs";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const ENGINE_DIR = dirname(fileURLToPath(import.meta.url));

function sourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    if (extname(entry.name) !== ".ts" || entry.name.endsWith(".test.ts")) return [];
    return [path];
  });
}

describe("certification engine architecture boundary", () => {
  it("does not depend on React, participant API, or rendering modules", () => {
    for (const path of sourceFiles(ENGINE_DIR)) {
      const source = readFileSync(path, "utf8");
      expect(source, path).not.toMatch(/from\s+["'](?:react|react-dom)["']/);
      expect(source, path).not.toMatch(/from\s+["']\.\.\/src(?:\/|["'])/);
      expect(source, path).not.toMatch(/from\s+["'][^"']*\/(?:api|components|features)(?:\/|["'])/);
    }
  });

  it("keeps worker execution on same-origin modules without dynamic code or shared memory", () => {
    for (const path of sourceFiles(ENGINE_DIR)) {
      const source = readFileSync(path, "utf8");
      expect(source, path).not.toMatch(/\beval\s*\(/);
      expect(source, path).not.toMatch(/\bnew\s+Function\b/);
      expect(source, path).not.toContain("SharedArrayBuffer");
      expect(source, path).not.toContain("URL.createObjectURL");
    }
  });
});
