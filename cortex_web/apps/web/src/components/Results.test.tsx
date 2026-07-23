import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { Results, type ResultSummary } from "./Results";

const SUMMARY: ResultSummary = {
  nQuestions: 159,
  stopReason: "all_estimated_or_undeterminable",
  verdicts: [
    "ABOVE_CUT", "INDETERMINATE_AT_CUT", "BELOW_CUT", "BELOW_CUT",
    "BELOW_CUT", "BELOW_CUT", "BELOW_CUT",
  ],
  terminationPolicy: "precision_v1",
  determinations: new Array(7).fill("DETERMINED"),
  taskCodes: ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"],
  taskLabels: ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"],
  taskClasses: ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"],
};

describe("participant results screen", () => {
  it("keeps the classification summary without AD6 technical details", () => {
    const html = renderToStaticMarkup(createElement(Results, { summary: SUMMARY }));
    expect(html).toContain("Assessment Complete");
    expect(html).toContain("Seizure");
    expect(html).toContain("AT EXPERT");
    expect(html).toContain("INDETERMINATE");
    expect(html).toContain("BELOW EXPERT");
    expect(html).not.toContain("technical details");
    expect(html).not.toContain("π (pass)");
    expect(html).not.toContain("info R");
  });
});
