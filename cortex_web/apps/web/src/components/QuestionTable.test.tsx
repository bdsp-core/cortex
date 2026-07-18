import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { QuestionRow } from "../api";
import { QuestionTable } from "./Shell";

describe("QuestionTable", () => {
  it("omits AD6-only diagnostics from participant history", () => {
    const rows: QuestionRow[] = [{
      q: 1,
      taskK: 2,
      domain: "IIIC",
      answer: "LPD",
      correct: "GPD",
      isCorrect: false,
      rt: 810,
      ell: -0.9,
      theta: -1.2,
    }];

    const html = renderToStaticMarkup(createElement(QuestionTable, { rows }));
    expect(html).toContain("Your answer");
    expect(html).toContain("ℓ / θ");
    expect(html).toContain("LPD");
    expect(html).toContain("GPD");
    expect(html).not.toContain("Δ info");
    expect(html).not.toContain("π");
    expect(html).not.toContain(">R</th>");
  });
});
