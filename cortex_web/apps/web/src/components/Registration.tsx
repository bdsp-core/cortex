// Registration wizard — 3 pages collecting the same participant record the
// desktop app stores (see scripts/cortex_storage.py _summary_row): identity,
// clinical background, demographics. Fields are de-identified/categorical;
// no PHI is required. The assembled record is attached to the session and
// surfaced in the mineable results summary.

import { useState } from "react";
import { COLORS, FONTS } from "../../ui/theme";
import { Button, Card, Field, Heading, SelectField, Stage } from "./ui";
import { CONSENT_VERSION, IRB_PROTOCOL_ID } from "./Consent";

export interface Participant {
  name: string;
  expertise: string;
  institution: string;
  practice_setting: string;
  years_reading_eeg: string;
  eeg_volume_per_month: string;
  self_rated_confidence: string;
  color_vision: string;
  prior_test_taken: string;
  sex: string;
  country: string;
  race_ethnicity: string;
  consent_version: string;
  irb_protocol_id: string;
}

const EXPERTISE = [
  "Attending epileptologist",
  "Attending neurologist (non-epilepsy)",
  "Clinical neurophysiology fellow",
  "Neurology resident",
  "EEG technologist",
  "Researcher / scientist",
  "Other",
];
const PRACTICE_SETTING = [
  "Academic medical center",
  "Community hospital",
  "Private practice",
  "Research institution",
  "Training program",
  "Other",
];
const YEARS = ["< 1", "1–2", "3–5", "6–10", "11–20", "> 20"];
const VOLUME = ["< 10", "10–50", "50–100", "100–250", "> 250"];
const CONFIDENCE = ["1 — Novice", "2", "3 — Competent", "4", "5 — Expert"];
const COLOR_VISION = ["Normal", "Color vision deficiency", "Unsure"];
const YESNO = ["No", "Yes"];
const SEX = ["Female", "Male", "Other", "Prefer not to say"];
const RACE = [
  "American Indian or Alaska Native",
  "Asian",
  "Black or African American",
  "Hispanic or Latino",
  "Native Hawaiian or Pacific Islander",
  "White",
  "Multiple / Other",
  "Prefer not to say",
];

const EMPTY: Participant = {
  name: "",
  expertise: "",
  institution: "",
  practice_setting: "",
  years_reading_eeg: "",
  eeg_volume_per_month: "",
  self_rated_confidence: "",
  color_vision: "",
  prior_test_taken: "",
  sex: "",
  country: "",
  race_ethnicity: "",
  consent_version: CONSENT_VERSION,
  irb_protocol_id: IRB_PROTOCOL_ID,
};

export function Registration({
  onComplete,
  onBack,
}: {
  onComplete: (p: Participant) => void;
  onBack: () => void;
}) {
  const [page, setPage] = useState(0);
  const [p, setP] = useState<Participant>(EMPTY);
  const set = (k: keyof Participant) => (v: string) => setP((prev) => ({ ...prev, [k]: v }));

  // page 0 requires identity + expertise; later pages are optional but encouraged.
  const page0Ok = p.name.trim() && p.expertise && p.institution.trim();

  const pages = [
    <div key="0">
      <Heading>Tell us about you</Heading>
      <Field label="Display name or initials" value={p.name} onChange={set("name")}
             required autoFocus placeholder="e.g. J.D." />
      <SelectField label="Primary role / expertise" value={p.expertise}
                   onChange={set("expertise")} options={EXPERTISE} required />
      <Field label="Institution" value={p.institution} onChange={set("institution")} required />
    </div>,
    <div key="1">
      <Heading>Clinical background</Heading>
      <SelectField label="Practice setting" value={p.practice_setting}
                   onChange={set("practice_setting")} options={PRACTICE_SETTING} />
      <SelectField label="Years reading EEG" value={p.years_reading_eeg}
                   onChange={set("years_reading_eeg")} options={YEARS} />
      <SelectField label="EEGs read per month" value={p.eeg_volume_per_month}
                   onChange={set("eeg_volume_per_month")} options={VOLUME} />
      <SelectField label="Self-rated IIIC confidence" value={p.self_rated_confidence}
                   onChange={set("self_rated_confidence")} options={CONFIDENCE} />
      <SelectField label="Have you taken this test before?" value={p.prior_test_taken}
                   onChange={set("prior_test_taken")} options={YESNO} />
    </div>,
    <div key="2">
      <Heading>Demographics (optional)</Heading>
      <p style={{ color: COLORS.textTertiary, fontSize: 13, marginTop: -8, marginBottom: 16 }}>
        Used only in aggregate for research; all fields optional.
      </p>
      <SelectField label="Color vision" value={p.color_vision}
                   onChange={set("color_vision")} options={COLOR_VISION} />
      <SelectField label="Sex" value={p.sex} onChange={set("sex")} options={SEX} />
      <Field label="Country" value={p.country} onChange={set("country")} />
      <SelectField label="Race / ethnicity" value={p.race_ethnicity}
                   onChange={set("race_ethnicity")} options={RACE} />
    </div>,
  ];

  return (
    <Stage maxW={620}>
      <Card>
        {/* step indicator */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          {pages.map((_, i) => (
            <div key={i} style={{
              flex: 1, height: 4, borderRadius: 2,
              background: i <= page ? COLORS.accent : COLORS.borderInactive,
            }} />
          ))}
        </div>
        {pages[page]}
        <div style={{ display: "flex", gap: 12, justifyContent: "space-between", marginTop: 16 }}>
          <Button kind="ghost" onClick={() => (page === 0 ? onBack() : setPage(page - 1))}>
            {page === 0 ? "Back" : "Previous"}
          </Button>
          {page < pages.length - 1 ? (
            <Button onClick={() => setPage(page + 1)} disabled={page === 0 && !page0Ok}>
              Next
            </Button>
          ) : (
            <Button onClick={() => onComplete(p)}>Continue to tutorial</Button>
          )}
        </div>
        <div style={{ fontFamily: FONTS.sans, color: COLORS.textTertiary, fontSize: 12,
                      marginTop: 16, textAlign: "center" }}>
          Step {page + 1} of {pages.length}
        </div>
      </Card>
    </Stage>
  );
}
