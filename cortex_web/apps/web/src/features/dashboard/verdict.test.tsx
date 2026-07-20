import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  VerdictChip,
  aurocOf,
  formatAssessmentDate,
  verdictPresentation,
  verdictsOf,
} from "./verdict";

describe("dashboard verdict presentation boundary", () => {
  it.each([
    ["ABOVE_CUT", "pass", "Above cut"],
    ["BELOW_CUT", "fail", "Below cut"],
    ["INDETERMINATE_AT_CUT", "referb", "Indeterminate at cut"],
    ["UNDETERMINABLE_CAP", "referu", "Insufficient precision · cap"],
    ["UNDETERMINABLE_BANK", "referu", "Insufficient precision · bank"],
  ])("maps %s without exposing engine diagnostics", (verdict, cls, text) => {
    expect(verdictPresentation(verdict)).toEqual({ cls, text });
    const html = renderToStaticMarkup(createElement(VerdictChip, { verdict }));
    expect(html).toContain(`cx-chip ${cls}`);
    expect(html).toContain(text);
    expect(html).not.toContain("pi");
  });

  it("reads legacy result fields defensively", () => {
    const result = { verdicts: ["PASS"], roc: [{ auroc: 0.91 }] };
    expect(verdictsOf(result)).toEqual(["PASS"]);
    expect(aurocOf(result, 0)).toBe(0.91);
    expect(aurocOf(result, 1)).toBeNull();
    expect(formatAssessmentDate(null)).toBe("—");
    expect(formatAssessmentDate("not-a-date")).toBe("not-a-date");
  });
});
