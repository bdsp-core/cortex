import { describe, expect, it } from "vitest";
import { reopenLabel, washoutActive } from "./washout";

const NOW = new Date("2026-07-11T08:00:00");

describe("reopenLabel", () => {
  it("same day: time only", () => {
    expect(reopenLabel("2026-07-11T18:30:00", NOW)).toMatch(/6:30/);
    expect(reopenLabel("2026-07-11T18:30:00", NOW)).not.toMatch(/tomorrow|on/);
  });
  it("next day: time + tomorrow", () => {
    expect(reopenLabel("2026-07-12T06:30:00", NOW)).toMatch(/tomorrow/);
  });
  it("later: time + date", () => {
    expect(reopenLabel("2026-07-14T06:30:00", NOW)).toMatch(/ on /);
  });
  it("garbage never crashes the banner", () => {
    expect(reopenLabel("not-a-date", NOW)).toBe("later");
  });
});

describe("washoutActive", () => {
  it("tracks the boundary", () => {
    expect(washoutActive("2026-07-11T09:00:00", NOW)).toBe(true);
    expect(washoutActive("2026-07-11T07:00:00", NOW)).toBe(false);
    expect(washoutActive(null, NOW)).toBe(false);
    expect(washoutActive("garbage", NOW)).toBe(false);
  });
});
