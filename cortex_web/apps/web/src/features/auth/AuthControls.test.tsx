import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AuthField, FormError, SignupSteps, scorePassword } from "./AuthControls";

describe("authentication control boundary", () => {
  it.each([
    ["", 0],
    ["longenough", 1],
    ["LongEnough", 2],
    ["LongEnough1", 3],
    ["LongEnough1!", 4],
  ])("scores password classes consistently", (password, expected) => {
    expect(scorePassword(password)).toBe(expected);
  });

  it("retains required accessible form and progress markup", () => {
    const field = renderToStaticMarkup(createElement(AuthField, {
      label: "Email",
      value: "reader@example.test",
      onChange: () => undefined,
      required: true,
      type: "email",
    }));
    expect(field).toContain("Email");
    expect(field).toContain("required");
    expect(field).toContain("reader@example.test");

    const steps = renderToStaticMarkup(createElement(SignupSteps, { on: 2 }));
    expect(steps.match(/<i/g)).toHaveLength(2);
    expect(renderToStaticMarkup(createElement(FormError, { children: null }))).toBe("");
  });
});
