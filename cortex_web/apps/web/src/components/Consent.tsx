// Consent screen — "Before You Begin". The participant must accept before the
// assessment starts. Declining returns to the landing screen. The accepted
// consent version + IRB protocol are recorded in the participant record.

import { COLORS } from "../../ui/theme";
import { Button, Card, Heading, Stage } from "./ui";

export const CONSENT_VERSION = "v1.0";
export const IRB_PROTOCOL_ID = "2016P000058 / 2013P001024";

export function Consent({ onAccept, onDecline }: { onAccept: () => void; onDecline: () => void }) {
  return (
    <Stage maxW={760}>
      <Card>
        <Heading>Before You Begin</Heading>
        <div style={{ color: COLORS.textBody, fontSize: 15, lineHeight: 1.6 }}>
          <p>
            This assessment measures your skill at classifying ictal–interictal–injury
            continuum (IIIC) EEG patterns. You will review a series of EEG recordings
            and, for each, select the single best label. The test adapts to your
            responses and ends automatically once your skill on every pattern is
            confidently determined.
          </p>
          <ul style={{ paddingLeft: 20 }}>
            <li>There is no time limit per recording, but answer as you would in practice.</li>
            <li>Your responses and de-identified background are recorded for research.</li>
            <li>No personally identifying information is required to participate.</li>
            <li>You may stop at any time by closing the window; partial data may be retained.</li>
          </ul>
          <p style={{ fontSize: 13, color: COLORS.textTertiary }}>
            Conducted under IRB {IRB_PROTOCOL_ID}. Consent version {CONSENT_VERSION}.
          </p>
        </div>
        <div style={{ display: "flex", gap: 12, justifyContent: "flex-end", marginTop: 24 }}>
          <Button kind="danger" onClick={onDecline}>Decline</Button>
          <Button onClick={onAccept}>I Accept</Button>
        </div>
      </Card>
    </Stage>
  );
}
