// Shared participant-profile field definitions. Single source of truth for the
// demographic/clinical questions collected at SIGNUP (AuthFlow) and editable in
// SETTINGS — so the two stay in sync and the pre-tutorial wizard is gone.
//
// The session record attached to a test (Participant) is assembled from the
// stored profile at test start (App.tsx); `name` = the account display name and
// the consent fields are constants added at session time.

import { CONSENT_VERSION, IRB_PROTOCOL_ID } from "./components/Consent";

export const EXPERTISE = [
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
const YEARS = ["< 1", "1-2", "3-5", "6-10", "11-20", "> 20"];
const VOLUME = ["< 10", "10-50", "50-100", "100-250", "> 250"];
// No em dashes (parenthetical anchors instead of "1 — Novice").
const CONFIDENCE = ["1 (Novice)", "2", "3 (Competent)", "4", "5 (Expert)"];
const COLOR_VISION = ["Normal", "Color vision deficiency", "Unsure"];
const YESNO = ["No", "Yes"];
const SEX = ["Female", "Male", "Other", "Prefer not to say"];
const AGE = ["<20", "20-40", "40-60", "60+"];

export interface ProfileFieldSpec {
  key: string;
  label: string;
  kind: "text" | "select";
  options?: readonly string[];
  placeholder?: string;
}

// The demographic sections shown AFTER the account step at signup, and as the
// editable profile form in Settings. (expertise lives on the account step.)
export const PROFILE_SECTIONS: { title: string; note?: string; fields: ProfileFieldSpec[] }[] = [
  {
    title: "Clinical background",
    fields: [
      { key: "institution", label: "Institution", kind: "text", placeholder: "e.g. Stanford" },
      { key: "practice_setting", label: "Practice setting", kind: "select", options: PRACTICE_SETTING },
      { key: "years_reading_eeg", label: "Years reading EEG", kind: "select", options: YEARS },
      { key: "eeg_volume_per_month", label: "EEGs read per month", kind: "select", options: VOLUME },
      { key: "self_rated_confidence", label: "Self-rated IIIC confidence", kind: "select", options: CONFIDENCE },
      { key: "prior_test_taken", label: "Have you taken this test before?", kind: "select", options: YESNO },
    ],
  },
  {
    title: "Demographics",
    fields: [
      { key: "age", label: "Age", kind: "select", options: AGE },
      { key: "sex", label: "Sex", kind: "select", options: SEX },
      { key: "location", label: "Location (city / region)", kind: "text", placeholder: "e.g. Boston, MA" },
      { key: "country", label: "Country", kind: "text", placeholder: "e.g. United States" },
      { key: "color_vision", label: "Color vision", kind: "select", options: COLOR_VISION },
    ],
  },
];

// Every profile key (excludes account-level fields: email, displayName).
export const PROFILE_KEYS: string[] = ["expertise",
  ...PROFILE_SECTIONS.flatMap((s) => s.fields.map((f) => f.key))];

export type Profile = Record<string, string>;

// The per-session participant record the engine/results expect (desktop parity).
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
  age: string;
  location: string;
  country: string;
  consent_version: string;
  irb_protocol_id: string;
}

// Build the session participant record from the stored account profile +
// display name. Consent constants are stamped at session time.
export function participantFromProfile(displayName: string, profile: Profile): Participant {
  const g = (k: string) => profile[k] ?? "";
  return {
    name: displayName,
    expertise: g("expertise"),
    institution: g("institution"),
    practice_setting: g("practice_setting"),
    years_reading_eeg: g("years_reading_eeg"),
    eeg_volume_per_month: g("eeg_volume_per_month"),
    self_rated_confidence: g("self_rated_confidence"),
    color_vision: g("color_vision"),
    prior_test_taken: g("prior_test_taken"),
    sex: g("sex"),
    age: g("age"),
    location: g("location"),
    country: g("country"),
    consent_version: CONSENT_VERSION,
    irb_protocol_id: IRB_PROTOCOL_ID,
  };
}
