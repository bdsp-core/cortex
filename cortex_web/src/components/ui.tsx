// Shared presentational primitives for the onboarding screens. Centralizes
// the palette/typography so every branding screen (Landing / Consent /
// Registration / Tutorial) is visually consistent with the desktop app.

import { CSSProperties, ReactNode } from "react";
import { COLORS, FONTS } from "../../ui/theme";

// Full-viewport dark stage, content centered. `maxW` reproduces the desktop's
// fixed window proportions as a responsive max-width.
export function Stage({ children, maxW = 980 }: { children: ReactNode; maxW?: number }) {
  return (
    <div
      style={{
        background: COLORS.bg,
        color: COLORS.textBody,
        fontFamily: FONTS.sans,
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        boxSizing: "border-box",
      }}
    >
      <div style={{ width: "100%", maxWidth: maxW }}>{children}</div>
    </div>
  );
}

// A raised card panel.
export function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div
      style={{
        background: COLORS.card,
        border: `1px solid ${COLORS.borderInactive}`,
        borderRadius: "var(--radius-panel)",
        padding: 32,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Wordmark({ size = 84 }: { size?: number }) {
  return (
    <div
      style={{
        fontFamily: FONTS.serif,
        fontSize: size,
        letterSpacing: size * 0.12,
        color: COLORS.textPrimary,
        textAlign: "center",
        fontWeight: 500,
      }}
    >
      CORTEX
    </div>
  );
}

export function Heading({ children, serif = true }: { children: ReactNode; serif?: boolean }) {
  return (
    <h1
      style={{
        fontFamily: serif ? FONTS.serif : FONTS.sans,
        color: COLORS.textPrimary,
        fontSize: 30,
        fontWeight: 500,
        margin: "0 0 16px",
      }}
    >
      {children}
    </h1>
  );
}

type BtnKind = "primary" | "ghost" | "danger";

export function Button({
  children,
  onClick,
  kind = "primary",
  disabled,
  type = "button",
  style,
}: {
  children: ReactNode;
  onClick?: () => void;
  kind?: BtnKind;
  disabled?: boolean;
  type?: "button" | "submit";
  style?: CSSProperties;
}) {
  const base: CSSProperties = {
    padding: "12px 24px",
    borderRadius: "var(--radius-ctl)",
    fontSize: 15,
    fontWeight: 600,
    fontFamily: FONTS.sans,
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.5 : 1,
    transition: "background 0.12s, border-color 0.12s",
    border: "1px solid transparent",
  };
  const kinds: Record<BtnKind, CSSProperties> = {
    primary: { background: COLORS.accent, color: "#1a1206", border: "none" },
    ghost: {
      background: "transparent",
      color: COLORS.textSecondary,
      border: `1px solid ${COLORS.borderInactive2}`,
    },
    danger: {
      background: "transparent",
      color: COLORS.fail,
      border: `1px solid ${COLORS.fail}`,
    },
  };
  return (
    <button type={type} onClick={onClick} disabled={disabled}
            style={{ ...base, ...kinds[kind], ...style }}>
      {children}
    </button>
  );
}

const labelStyle: CSSProperties = {
  display: "block",
  fontSize: 13,
  color: COLORS.textBody,
  marginBottom: 6,
  fontWeight: 600,
};
const inputStyle: CSSProperties = {
  width: "100%",
  padding: "10px 12px",
  borderRadius: "var(--radius-ctl)",
  background: COLORS.cardAlt,
  border: `1px solid ${COLORS.borderInactive2}`,
  color: COLORS.textPrimary,
  fontSize: 14,
  fontFamily: FONTS.sans,
  boxSizing: "border-box",
};

export function Field({
  label,
  value,
  onChange,
  type = "text",
  placeholder,
  required,
  autoFocus,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
  required?: boolean;
  autoFocus?: boolean;
}) {
  return (
    <label style={{ display: "block", marginBottom: 16 }}>
      <span style={labelStyle}>
        {label}
        {required && <span style={{ color: COLORS.accent }}> *</span>}
      </span>
      <input
        type={type}
        value={value}
        placeholder={placeholder}
        autoFocus={autoFocus}
        onChange={(e) => onChange(e.target.value)}
        style={inputStyle}
      />
    </label>
  );
}

export function SelectField({
  label,
  value,
  onChange,
  options,
  required,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  required?: boolean;
}) {
  return (
    <label style={{ display: "block", marginBottom: 16 }}>
      <span style={labelStyle}>
        {label}
        {required && <span style={{ color: COLORS.accent }}> *</span>}
      </span>
      <select value={value} onChange={(e) => onChange(e.target.value)} style={inputStyle}>
        <option value="">— select —</option>
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </label>
  );
}

export function ErrorText({ children }: { children: ReactNode }) {
  if (!children) return null;
  return (
    <div style={{ color: COLORS.fail, fontSize: 13, marginTop: 8 }}>{children}</div>
  );
}
