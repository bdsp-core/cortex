// Public Privacy Policy page, served at /privacy (Caddy try_files → index.html;
// main.tsx renders this instead of <App/> when the path is /privacy). Mirrors
// the auth surface aesthetic: themed CSS vars, horizontal wordmark, theme
// toggle, teal accents, sharp panel radii. Content is the PUBLIC policy only
// (the internal data-protection analysis in docs/PRIVACY_POLICY_DRAFT.md is
// never shipped). Keep this in sync with that file's Part 2 if the policy
// changes.

import { CSSProperties, ReactNode } from "react";
import { COLORS, FONTS } from "../../ui/theme";
import { ThemeToggle } from "../theme/ThemeProvider";

const EFFECTIVE_DATE = "June 18, 2026";

const wrap: CSSProperties = {
  minHeight: "100vh",
  background: COLORS.bg,
  color: COLORS.textBody,
  fontFamily: FONTS.sans,
  boxSizing: "border-box",
  position: "relative",
  padding: "40px 24px 72px",
};
const column: CSSProperties = {
  width: "100%",
  maxWidth: 760,
  margin: "0 auto",
};
const titleStyle: CSSProperties = {
  fontFamily: FONTS.sans,
  fontSize: 30,
  fontWeight: 700,
  letterSpacing: "-0.01em",
  color: COLORS.textPrimary,
  margin: "0 0 4px",
};
const effStyle: CSSProperties = { fontSize: 13, color: COLORS.textFaint, margin: "0 0 8px" };
const h2Style: CSSProperties = {
  fontFamily: FONTS.sans,
  fontSize: 18,
  fontWeight: 600,
  color: COLORS.textPrimary,
  margin: "32px 0 10px",
  paddingTop: 20,
  borderTop: `1px solid ${COLORS.borderInactive}`,
};
const pStyle: CSSProperties = { fontSize: 15, lineHeight: 1.7, color: COLORS.textBody, margin: "0 0 14px" };
const liStyle: CSSProperties = { fontSize: 15, lineHeight: 1.7, color: COLORS.textBody, marginBottom: 6 };
const linkStyle: CSSProperties = { color: "var(--teal-deep)", fontWeight: 600, textDecoration: "none" };
const noteStyle: CSSProperties = {
  fontSize: 14,
  lineHeight: 1.7,
  color: COLORS.textBody,
  background: COLORS.cardAlt,
  borderLeft: "3px solid var(--teal)",
  borderRadius: "var(--radius-ctl)",
  padding: "14px 16px",
  margin: "0 0 14px",
};

function P({ children }: { children: ReactNode }) { return <p style={pStyle}>{children}</p>; }
function H2({ children }: { children: ReactNode }) { return <h2 style={h2Style}>{children}</h2>; }
function Mail({ a }: { a: string }) { return <a style={linkStyle} href={`mailto:${a}`}>{a}</a>; }

export function PrivacyPage() {
  return (
    <div style={wrap}>
      <ThemeToggle style={{ position: "absolute", top: 24, right: 24 }} />
      <div style={column}>
        <a href="/" aria-label="Back to CORTEX" style={{ display: "inline-block", marginBottom: 28 }}>
          <img
            src="/cortex_logo_word_horizontal@3x.png"
            srcSet="/cortex_logo_word_horizontal@2x.png 2x, /cortex_logo_word_horizontal@3x.png 3x"
            alt="CORTEX — EEG Skill Certification"
            style={{ height: 34, width: "auto", display: "block" }}
          />
        </a>

        <h1 style={titleStyle}>Privacy Policy</h1>
        <p style={effStyle}>Effective date: {EFFECTIVE_DATE}</p>

        <H2>Who we are</H2>
        <P>
          CORTEX is an online platform that helps neurologists and other EEG readers assess and
          improve their skill at recognizing patterns in EEG recordings. This Privacy Policy explains
          what personal data we collect when you use CORTEX at app.cortexeeg.org, why we collect it,
          how we protect it, and the choices and rights you have.
        </P>
        <P>
          CORTEX is part of the Westover Lab at Stanford University School of Medicine and is sponsored
          by the Clinical Data Animation Center (CDAC). The data controller is Stanford University
          School of Medicine (Westover Lab — CORTEX project), 453 Quarry Rd, Office 221B, Palo Alto,
          CA 94304. If you have any questions, contact us at <Mail a="mbwest@stanford.edu" /> or{" "}
          <Mail a="ewk23@stanford.edu" />. The individual responsible for data protection for CORTEX is
          Michael B. Westover, MD, PhD, Professor of Neurology, Stanford University School of Medicine;
          privacy and legal matters are overseen by the legal advisors to Stanford University School of
          Medicine.
        </P>
        <P>
          This policy covers your personal data as a <strong>user</strong> of CORTEX. The EEG examples
          you review in the assessment are pre-existing, de-identified research recordings; they are not
          your patients' data and are not collected from you.
        </P>
        <div style={noteStyle}>
          <strong>A note on research participation.</strong> CORTEX is used in connection with a
          research study. Separately from this Privacy Policy, you are shown a research informed-consent
          notice before your first assessment. That consent governs your participation in the research;
          this Privacy Policy explains how we handle your personal data. The Principal Investigator is
          Michael B. Westover, MD, PhD, Professor of Neurology, Stanford University School of Medicine
          (<Mail a="mbwest@stanford.edu" />).
        </div>

        <H2>The data we collect</H2>
        <P>
          <strong>Account information.</strong> Your email address, the display name you choose, and an
          optional self-reported professional role/expertise. If you create your account with a
          password, we store a securely hashed version of your password (never the password itself). If
          you use "Sign in with Google," we receive your email address, name, and a Google account
          identifier from Google (see below).
        </P>
        <P>
          <strong>Assessment and progress data.</strong> Your responses during assessments, whether
          they were correct, how long they took, and the results we generate from them, as well as
          practice/training activity and progress over time. We use this to show you your results and to
          support the research purpose described above. We describe what this data is used for, not the
          proprietary methods we use to analyze it.
        </P>
        <P>
          <strong>Technical and usage data.</strong> To keep the service secure and working, we record
          limited technical information such as the IP address used at sign-up and temporary counters
          used to prevent abuse (for example, limiting repeated sign-in attempts).
        </P>
        <P>
          <strong>Cookies and local storage.</strong> After you sign in, we store a temporary sign-in
          token in your browser's local storage so you stay logged in for a limited time. This is
          strictly functional. We do not use advertising cookies or third-party tracking, and we do not
          track you across other websites.
        </P>

        <H2>How and why we use your data</H2>
        <ul>
          <li style={liStyle}>To create and secure your account and sign you in.</li>
          <li style={liStyle}>To send you account emails (such as a verification code or a password-reset code).</li>
          <li style={liStyle}>To deliver the assessment, generate and show your results, and provide practice and progress features.</li>
          <li style={liStyle}>To support the research purpose for which CORTEX is used.</li>
          <li style={liStyle}>To protect the service against fraud, abuse, and security threats.</li>
          <li style={liStyle}>To comply with our legal obligations.</li>
        </ul>

        <H2>Legal bases (for users in the EEA, UK, and similar regimes)</H2>
        <P>
          We rely on: <strong>performance of a contract</strong> (to provide the account and assessment
          you request); <strong>your consent</strong> (for participation in the research purpose; you
          may withdraw it at any time); and our <strong>legitimate interests</strong> (to keep the
          service secure and prevent abuse). Where we rely on consent, withdrawing it will not affect the
          lawfulness of processing before withdrawal.
        </P>

        <H2>Sign in with Google</H2>
        <P>
          If you choose "Continue with Google," Google handles signing you in and sends us your email
          address, name, and a Google account identifier so we can create or access your CORTEX account.
          We use these only for authentication and your account profile. We do <strong>not</strong>{" "}
          receive your Google password, and we do not access your Gmail, contacts, or other Google data.
          Google's handling of your information is governed by Google's own privacy policy.
        </P>

        <H2>Service providers and international transfers</H2>
        <P>We use trusted service providers to run CORTEX, who process data only on our instructions:</P>
        <ul>
          <li style={liStyle}>
            <strong>Amazon Web Services</strong> — hosting and database, and email delivery for account
            messages. Our servers and database are located in the United States (Oregon).
          </li>
          <li style={liStyle}><strong>Box</strong> — encrypted, access-controlled backups.</li>
          <li style={liStyle}><strong>Google</strong> — "Sign in with Google" authentication, if you choose it.</li>
        </ul>
        <P>
          Because our infrastructure is in the United States, if you access CORTEX from outside the US
          your personal data will be transferred to and stored in the US. Where required, we rely on
          appropriate safeguards for these transfers, including Standard Contractual Clauses with our
          providers.
        </P>

        <H2>How long we keep your data</H2>
        <P>
          We keep your account and assessment data for as long as your account is active. You can ask us
          to delete your data at any time (see "Your rights and choices"); when you do, we will delete
          it, unless a longer period is required by law. Short-lived items (such as email verification
          codes) expire within minutes. Backups are kept for up to 30 days and then deleted, so it can
          take up to 30 days for a deletion to fully propagate out of backups.
        </P>

        <H2>How we protect your data</H2>
        <P>
          We use industry-standard safeguards, including encryption in transit (HTTPS), strong one-way
          hashing of passwords, protection of verification codes at rest, time-limited sign-in tokens,
          access controls, rate limiting, and encrypted off-site backups. No system is perfectly secure,
          but we work to protect your data and to limit what we collect to what we need.
        </P>

        <H2>Your rights and choices</H2>
        <P>
          Depending on where you live, you may have the right to: <strong>access</strong> a copy of your
          data; <strong>correct</strong> inaccurate data; <strong>delete</strong> your data;{" "}
          <strong>restrict</strong> or <strong>object</strong> to certain processing; receive your data
          in a <strong>portable</strong> format; and <strong>withdraw consent</strong> or{" "}
          <strong>withdraw from the research</strong> at any time. We do not sell or share your personal
          information, and we do not use it for targeted advertising.
        </P>
        <P>
          To exercise any of these rights — including to access or delete your data — email us at{" "}
          <Mail a="mbwest@stanford.edu" /> or <Mail a="ewk23@stanford.edu" />. We will respond within the
          time required by applicable law. If you are in the EEA or UK, you also have the right to lodge a
          complaint with your local data-protection authority.
        </P>

        <H2>Children</H2>
        <P>
          CORTEX is intended for licensed clinicians and trainees and is not directed to anyone under 18.
          We do not knowingly collect data from children.
        </P>

        <H2>Changes to this policy</H2>
        <P>
          We may update this policy from time to time. We will post the updated version here and change
          the "Effective date" above; significant changes will be communicated as required by law.
        </P>

        <H2>Contact us</H2>
        <P>
          Stanford University School of Medicine — Westover Lab (CORTEX project), sponsored by the
          Clinical Data Animation Center (CDAC)<br />
          453 Quarry Rd, Office 221B, Palo Alto, CA 94304<br />
          <Mail a="mbwest@stanford.edu" /> · <Mail a="ewk23@stanford.edu" />
        </P>
        <P>This policy is governed by the laws of the State of California, United States.</P>

        <div style={{ marginTop: 40, paddingTop: 20, borderTop: `1px solid ${COLORS.borderInactive}` }}>
          <a href="/" style={linkStyle}>← Back to CORTEX</a>
        </div>
      </div>
    </div>
  );
}
