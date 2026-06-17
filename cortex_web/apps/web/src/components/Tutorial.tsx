// Tutorial screen — explains the viewer layout, controls, and how the test
// ends, before the assessment begins. Keyboard-driven answering is the key
// thing to convey (1–6 selects AND advances).

import { COLORS, FONTS, IIIC_OPTIONS } from "../../ui/theme";
import { Button, Card, Heading, Stage } from "./ui";

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px", borderRadius: 4,
      background: COLORS.cardAlt, border: `1px solid ${COLORS.borderInactive2}`,
      color: COLORS.textPrimary, fontFamily: "monospace", fontSize: 13, margin: "0 2px",
    }}>{children}</span>
  );
}

export function Tutorial({ onStart, onBack }: { onStart: () => void; onBack: () => void }) {
  return (
    <Stage maxW={820}>
      <Card>
        <Heading>How the assessment works</Heading>
        <div style={{ color: COLORS.textBody, fontSize: 15, lineHeight: 1.6 }}>
          <p>
            Each screen shows a 30-second EEG segment (right) and its 10-minute
            spectrogram (left). A dashed white line on the spectrogram marks where
            the displayed EEG sits in time. Choose the single best IIIC label:
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, margin: "12px 0" }}>
            {IIIC_OPTIONS.map((o, i) => (
              <span key={o.code} style={{
                padding: "6px 12px", borderRadius: 6, background: COLORS.cardAlt,
                border: `1px solid ${COLORS.borderInactive2}`, color: COLORS.textSecondary,
                fontSize: 14,
              }}>
                <b style={{ color: COLORS.accent }}>{i + 1}</b> · {o.label}
              </span>
            ))}
          </div>
          <p>
            Press <Kbd>1</Kbd>–<Kbd>6</Kbd> to select a label — your choice is
            recorded and the next recording loads immediately (no confirm step).
            You can also click a button.
          </p>
          <p style={{ marginBottom: 4 }}>Adjust the display anytime:</p>
          <ul style={{ paddingLeft: 20, marginTop: 4 }}>
            <li><Kbd>←</Kbd> <Kbd>→</Kbd> pan the EEG · <Kbd>↑</Kbd> <Kbd>↓</Kbd> change gain · <Kbd>Ctrl</Kbd> cycle montage</li>
            <li>Dropdowns for montage, gain, band-pass, notch, and window length</li>
          </ul>
          <p>
            The test is <b>adaptive</b>: it selects the most informative recording
            for you each step and ends automatically once your skill on every
            pattern is confidently determined. A counter shows how many questions
            you've answered; most participants finish well before the maximum.
          </p>
        </div>
        <div style={{ display: "flex", gap: 12, justifyContent: "space-between", marginTop: 24 }}>
          <Button kind="ghost" onClick={onBack}>Back</Button>
          <Button onClick={onStart} style={{ fontFamily: FONTS.sans }}>Begin the test</Button>
        </div>
      </Card>
    </Stage>
  );
}
