import { describe, expect, it } from "vitest";
import { isPhoneDevice } from "./device";

describe("isPhoneDevice", () => {
  it("matches a phone in both orientations (short side < 768, coarse pointer)", () => {
    expect(isPhoneDevice("", 390, true)).toBe(true);   // portrait
    expect(isPhoneDevice("", 390, true)).toBe(true);   // landscape (short side unchanged)
  });

  it("keeps tablets and desktops on the desktop-layout web app", () => {
    expect(isPhoneDevice("", 768, true)).toBe(false);  // iPad short side
    expect(isPhoneDevice("", 1080, false)).toBe(false); // desktop
    expect(isPhoneDevice("", 500, false)).toBe(false); // narrow desktop window, fine pointer
  });

  it("honors the overrides", () => {
    expect(isPhoneDevice("?desktop=1", 390, true)).toBe(false);
    expect(isPhoneDevice("?mobile=1", 1080, false)).toBe(true);
    // desktop=1 wins when both are present (the safety hatch)
    expect(isPhoneDevice("?desktop=1&mobile=1", 390, true)).toBe(false);
  });
});
