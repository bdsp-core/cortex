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

  it("renders every graded bias-flag tier and nothing for withheld/unknown flags", () => {
    // Six graded states across six domains; a withheld (null) flag and an
    // unknown future string must render nothing (BIAS_FLAG_LABEL miss).
    const flagged: ResultSummary = {
      ...SUMMARY,
      biasFlags: [
        "WATCH_OVERCALLER", "WATCH_UNDERCALLER",
        "EXTREME_OVERCALLER", "EXTREME_UNDERCALLER",
        "EXTREME_CONFIRMED_OVERCALLER", "EXTREME_CONFIRMED_UNDERCALLER",
        null,
      ],
    };
    const html = renderToStaticMarkup(createElement(Results, { summary: flagged }));
    expect(html).toContain("mild tendency to over-report");
    expect(html).toContain("mild tendency to under-report");
    expect(html).toContain("Calibration note: tendency to over-report");
    expect(html).toContain("Calibration note: tendency to under-report");
    expect(html).toContain("consistent, strongly supported tendency to over-report");
    expect(html).toContain("consistent, strongly supported tendency to under-report");
    // Six calibration notes exactly: none for the withheld seventh domain.
    expect(html.match(/Calibration note:/g)).toHaveLength(6);

    const unknown: ResultSummary = {
      ...SUMMARY, biasFlags: ["SOME_FUTURE_FLAG", null, null, null, null, null, null],
    };
    expect(renderToStaticMarkup(createElement(Results, { summary: unknown })))
      .not.toContain("Calibration note:");
  });
});
