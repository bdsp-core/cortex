import { describe, expect, it } from "vitest";
import {
  parseAuthDeepLink,
  parseAuthStartScreen,
  parseCohortDeepLink,
} from "./deepLink";

describe("parseCohortDeepLink", () => {
  it("parses a cohort-invite link", () => {
    expect(parseCohortDeepLink("?cohort=ch-AbC123_xy-9")).toBe("ch-AbC123_xy-9");
  });
  it("rejects malformed ids and unrelated queries", () => {
    expect(parseCohortDeepLink("?cohort=nope")).toBeNull();
    expect(parseCohortDeepLink("?cohort=ch-a")).toBeNull();
    expect(parseCohortDeepLink("")).toBeNull();
  });
});

describe("parseAuthDeepLink", () => {
  it("parses a verify link", () => {
    expect(parseAuthDeepLink("?verifyEmail=a%40b.org&verifyCode=123456"))
      .toEqual({ kind: "verify", email: "a@b.org", code: "123456" });
  });

  it("parses a reset link", () => {
    expect(parseAuthDeepLink("?resetEmail=x%2By%40z.edu&resetCode=000042"))
      .toEqual({ kind: "reset", email: "x+y@z.edu", code: "000042" });
  });

  it("normalizes the email case", () => {
    expect(parseAuthDeepLink("?verifyEmail=A%40B.ORG&verifyCode=123456")?.email)
      .toBe("a@b.org");
  });

  it("rejects malformed codes", () => {
    expect(parseAuthDeepLink("?verifyEmail=a%40b.org&verifyCode=12345")).toBeNull();
    expect(parseAuthDeepLink("?verifyEmail=a%40b.org&verifyCode=12345x")).toBeNull();
    expect(parseAuthDeepLink("?verifyEmail=a%40b.org")).toBeNull();
  });

  it("rejects a non-email", () => {
    expect(parseAuthDeepLink("?verifyEmail=nope&verifyCode=123456")).toBeNull();
  });

  it("is quiet on unrelated or empty queries", () => {
    expect(parseAuthDeepLink("")).toBeNull();
    expect(parseAuthDeepLink("?utm_source=x")).toBeNull();
  });
});

describe("parseAuthStartScreen", () => {
  it("opens account creation only for the declared public route", () => {
    expect(parseAuthStartScreen("?auth=signup")).toBe("signup");
    expect(parseAuthStartScreen("?cohort=ch-abcdef&auth=signup")).toBe("signup");
  });

  it("ignores unknown or missing auth screens", () => {
    expect(parseAuthStartScreen("?auth=signin")).toBeNull();
    expect(parseAuthStartScreen("?auth=unknown")).toBeNull();
    expect(parseAuthStartScreen("")).toBeNull();
  });
});
