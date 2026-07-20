import { type CSSProperties, type ReactNode, useState } from "react";

import { COLORS, FONTS } from "../../../ui/theme";

export const authLabelStyle: CSSProperties = {
  display: "block",
  fontSize: 12,
  fontWeight: 600,
  color: COLORS.textBody,
  marginBottom: 4,
  letterSpacing: "0.02em",
};

export const authInputStyle: CSSProperties = {
  width: "100%",
  padding: "12px 12px",
  border: `1px solid ${COLORS.borderInactive2}`,
  borderRadius: "var(--radius-ctl)",
  background: COLORS.cardAlt,
  color: COLORS.textPrimary,
  fontFamily: FONTS.sans,
  fontSize: 14,
  lineHeight: 1.3,
  boxSizing: "border-box",
};

type ButtonVariant = "primary" | "secondary" | "accentOutline";

export function AuthButton({
  children,
  variant = "primary",
  type = "button",
  disabled,
  onClick,
  style,
}: {
  children: ReactNode;
  variant?: ButtonVariant;
  type?: "button" | "submit";
  disabled?: boolean;
  onClick?: () => void;
  style?: CSSProperties;
}) {
  const [hover, setHover] = useState(false);
  const activeHover = hover && !disabled;
  const base: CSSProperties = {
    fontFamily: FONTS.sans,
    fontSize: 14,
    fontWeight: 600,
    padding: "12px 16px",
    borderRadius: "var(--radius-ctl)",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.45 : 1,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    boxSizing: "border-box",
    transition: "background 0.12s, border-color 0.12s, color 0.12s",
  };
  const variants: Record<ButtonVariant, CSSProperties> = {
    primary: {
      background: activeHover ? "var(--teal-hover)" : "var(--teal)",
      border: `1px solid ${activeHover ? "var(--teal-hover)" : "var(--teal)"}`,
      color: "#fff",
    },
    secondary: {
      background: activeHover ? "var(--btn-hover)" : COLORS.card,
      border: `1px solid ${COLORS.borderInactive2}`,
      color: COLORS.textPrimary,
    },
    accentOutline: {
      background: activeHover ? "var(--btn-hover)" : COLORS.card,
      border: "1px solid var(--teal)",
      color: "var(--teal-deep)",
    },
  };
  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{ ...base, ...variants[variant], ...style }}
    >
      {children}
    </button>
  );
}

export function AuthField({
  label,
  type = "text",
  value,
  onChange,
  placeholder,
  required,
  autoFocus,
  autoComplete,
  onInput,
  children,
  name,
}: {
  label?: ReactNode;
  type?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  required?: boolean;
  autoFocus?: boolean;
  autoComplete?: string;
  onInput?: (value: string) => void;
  children?: ReactNode;
  name?: string;
}) {
  return (
    <label style={{ display: "block", marginBottom: 16 }}>
      {label && <span style={authLabelStyle}>{label}</span>}
      <input
        type={type}
        name={name}
        value={value}
        placeholder={placeholder}
        required={required}
        autoFocus={autoFocus}
        autoComplete={autoComplete}
        onChange={(event) => {
          onChange(event.target.value);
          onInput?.(event.target.value);
        }}
        style={authInputStyle}
      />
      {children}
    </label>
  );
}

export function FormError({ children }: { children: ReactNode }) {
  if (!children) return null;
  return <div style={{ fontSize: 13, color: COLORS.fail, marginTop: 12 }}>{children}</div>;
}

export function SignupSteps({ on }: { on: 1 | 2 }) {
  return (
    <div style={{ display: "flex", gap: 8, marginBottom: 24 }}>
      {[1, 2].map((step) => (
        <i key={step} style={{
          flex: 1,
          height: 3,
          borderRadius: 0,
          background: step <= on ? "var(--teal)" : COLORS.borderInactive,
        }} />
      ))}
    </div>
  );
}

export function scorePassword(value: string): number {
  let score = 0;
  if (value.length >= 8) score += 1;
  if (/[A-Z]/.test(value) && /[a-z]/.test(value)) score += 1;
  if (/[0-9]/.test(value)) score += 1;
  if (/[^A-Za-z0-9]/.test(value)) score += 1;
  return score;
}

export function PasswordStrength({ score }: { score: number }) {
  const colorFor = (index: number) => {
    if (index >= score) return COLORS.borderInactive;
    if (score === 1) return COLORS.fail;
    if (score === 2) return "var(--refer-b)";
    if (score === 3) return "var(--teal-mid)";
    return "var(--teal)";
  };
  return (
    <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
      {[0, 1, 2, 3].map((index) => (
        <i key={index} style={{
          flex: 1,
          height: 3,
          background: colorFor(index),
          transition: "background 0.2s",
        }} />
      ))}
    </div>
  );
}
