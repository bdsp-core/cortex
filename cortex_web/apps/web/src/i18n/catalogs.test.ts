// Catalog copy conventions. House style bans em dashes in user-facing
// messaging (see profileFields.ts: "No em dashes"); rephrase with a period,
// semicolon, or comma instead. Placeholder glyphs for missing VALUES ("—" in
// tables) live in components, not catalogs, so a catalog hit is always prose.

import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const localesDir = join(dirname(fileURLToPath(import.meta.url)), "locales");

describe("i18n catalogs", () => {
  it("contain no em dashes (house messaging style)", () => {
    const files = readdirSync(localesDir).filter((f) => f.endsWith(".json"));
    expect(files.length).toBeGreaterThanOrEqual(8);
    for (const f of files) {
      const catalog = JSON.parse(readFileSync(join(localesDir, f), "utf8"));
      for (const [key, value] of Object.entries(catalog)) {
        expect(String(value).includes("—"),
          `${f} ${key} contains an em dash`).toBe(false);
      }
    }
  });
});
