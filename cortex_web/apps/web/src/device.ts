// Boot-time phone detection: the phone companion surface (src/mobile/) is a
// SEPARATE module tree from the desktop app, selected ONCE at boot in
// main.tsx — deliberately no live desktop↔mobile switching on resize (it
// would drop in-flight state, and a certification sitting must never change
// modality mid-test).
//
// "Phone" = coarse pointer AND a screen whose short side is under 768 CSS px:
// phones match in either orientation; iPads and laptops (fine pointer, or
// short side ≥ 768) stay on the desktop app; a narrowed desktop window stays
// desktop. ?desktop=1 / ?mobile=1 are the overrides for testing and as the
// escape hatch.

export function isPhoneDevice(
  search: string, minScreenDim: number, coarsePointer: boolean,
): boolean {
  const p = new URLSearchParams(search);
  if (p.get("desktop") === "1") return false;
  if (p.get("mobile") === "1") return true;
  return coarsePointer && minScreenDim < 768;
}

export function detectPhone(): boolean {
  return isPhoneDevice(
    window.location.search,
    Math.min(window.screen.width, window.screen.height),
    window.matchMedia("(pointer: coarse)").matches,
  );
}
