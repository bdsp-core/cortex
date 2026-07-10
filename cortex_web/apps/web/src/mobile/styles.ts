// Style primitives for the phone companion surface — single-column, thumb-
// sized touch targets, same brand tokens as desktop (ui/theme). Kept local to
// src/mobile/ so phone layout decisions never leak into desktop components.

import { CSSProperties } from "react";
import { COLORS, FONTS } from "../../ui/theme";

export const page: CSSProperties = {
  minHeight: "100dvh",
  background: COLORS.bg,
  fontFamily: FONTS.sans,
  color: COLORS.textPrimary,
  display: "flex",
  flexDirection: "column",
};

export const header: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "14px 16px",
  borderBottom: `1px solid ${COLORS.borderInactive2}`,
};

export const main: CSSProperties = {
  flex: 1,
  padding: 16,
  display: "flex",
  flexDirection: "column",
  gap: 16,
  maxWidth: 560,             // phones + small foldables; centered if wider
  width: "100%",
  margin: "0 auto",
  boxSizing: "border-box",
};

export const card: CSSProperties = {
  background: COLORS.card,
  border: `1px solid ${COLORS.borderInactive}`,
  borderRadius: "var(--radius-panel)",
  padding: 16,
};

export const sectionTitle: CSSProperties = {
  fontSize: 13,
  fontWeight: 700,
  letterSpacing: 0.4,
  textTransform: "uppercase",
  color: COLORS.textFaint,
  margin: "0 0 10px",
};

export const rowLine: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 12,
  padding: "10px 0",
  borderTop: `1px solid ${COLORS.borderInactive}`,
  fontSize: 14,
};

export const faint: CSSProperties = { color: COLORS.textFaint, fontSize: 12 };

export const ghostBtn: CSSProperties = {
  background: "none",
  border: `1px solid ${COLORS.borderInactive2}`,
  borderRadius: 8,
  color: COLORS.textBody,
  fontFamily: FONTS.sans,
  fontSize: 14,
  padding: "10px 14px",     // ≥44px touch target with line height
  cursor: "pointer",
};

export const primaryBtn: CSSProperties = {
  ...ghostBtn,
  background: "var(--teal)",
  border: "1px solid var(--teal)",
  color: "#fff",
  fontWeight: 600,
};

export const label: CSSProperties = {
  display: "block",
  fontSize: 12,
  fontWeight: 600,
  color: COLORS.textFaint,
  margin: "12px 0 4px",
};

export const input: CSSProperties = {
  width: "100%",
  boxSizing: "border-box",
  fontSize: 16,             // <16px makes iOS Safari zoom the page on focus
  fontFamily: FONTS.sans,
  color: COLORS.textPrimary,
  background: COLORS.cardAlt,
  border: `1px solid ${COLORS.borderInactive2}`,
  borderRadius: 8,
  padding: "10px 12px",
};
