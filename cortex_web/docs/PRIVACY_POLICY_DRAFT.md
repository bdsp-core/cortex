> **DRAFT — for review by the CORTEX team, the PI, and qualified legal counsel.**
> This document is a working draft prepared to support a privacy program and the
> "Sign in with Google" launch. It is **not legal advice** and must be reviewed
> and approved by counsel admitted in the relevant jurisdiction(s) before it is
> published or relied upon. Team-supplied details were filled in on 2026-06-18;
> two items remain for counsel to confirm (see Part 1 §1.6 item 7).

---

# PART 1 — Internal Data-Protection Analysis (not for publication)

Prepared as data-protection counsel; every data-handling claim below was checked
against the application source (`services/api/db.py`, `app.py`, `security.py`,
`mailer.py`, `deploy/scripts/backup_to_box.sh`). Where the policy in Part 2 is
deliberately vague to protect intellectual property, this Part is precise.

## 1.1 Framing: whose data is this?

CORTEX processes personal data **about the clinician-user** (identity +
assessment performance). The EEG signals shown during the assessment are
**pre-existing, de-identified research data** curated from an external source;
they are **not** collected from the user and are not the user's patients' data.
This matters legally:

- The privacy policy governs the **user's** personal data only.
- It is **separate** from, and additional to, the **research-study / IRB
  informed consent** the application already presents before a user's first
  assessment. The privacy policy should not be drafted as, or substituted for,
  that consent.
- No patient PHI is collected *from the user*, so this is not a HIPAA
  covered-entity data flow on the user side. (The provenance/DUA obligations for
  the underlying de-identified EEG corpus are governed elsewhere and are out of
  scope for this user-facing policy.)

## 1.2 Data inventory (grounded in the schema)

| Category | Fields / source | Purpose | Storage | Retention (as-built) | Security control |
|---|---|---|---|---|---|
| Account identity | `participants.email`, `display_name`, self-reported `expertise` (collected at signup; **not currently persisted** in `register_participant`) | Create/identify the account, address the user | Postgres on EC2 (AWS us-west-2, Oregon) | Indefinite — **no deletion mechanism in code** | TLS in transit; DB on private host |
| Password (password accounts) | `participants.password_hash` | Authenticate | Postgres | Indefinite | **PBKDF2-HMAC-SHA256, 200k iters, 16-byte salt** — not reversible |
| Google identity (planned) | `google_sub`, email, name from the Google ID token | Authenticate via "Sign in with Google" | Postgres | Indefinite | Google-signed token verified server-side; no Google password ever seen |
| Email-verification & reset codes | `auth_codes.code_hash`, `purpose`, timestamps, `attempts` | Verify email ownership / reset password | Postgres | Short-lived (15-min TTL; one active row per purpose) | **HMAC-SHA256 at rest** (peppered with server secret); 5-attempt cap |
| Session token | HS256 JWT (`sub`=internal code, 6-hour `exp`) | Keep the user signed in | **Client-side `localStorage`** only | 6 hours | Signed with server secret; expiry enforced on decode |
| Network/technical | `participants.signup_ip` (from `X-Forwarded-For`); transient rate-limit counters keyed by IP | Abuse/fraud prevention, signup integrity | Postgres (`signup_ip`); in-memory (rate limiter) | `signup_ip` indefinite | TLS; not exposed via any read endpoint |
| Assessment performance | `sessions` (timing, status, counts), `trials` (per-item response, correctness, reaction time, item id), `results` (scored result JSON) | Deliver the assessment + the user's results/dashboard; aggregate research | Postgres | Indefinite | TLS; gated behind the user's JWT |
| Training/progress | `regimens`, `training_sessions`, `param_trajectories` | Personalized practice + progress charts | Postgres | Indefinite | TLS; JWT-gated |

## 1.3 Processors / sub-processors

- **Amazon Web Services (AWS)** — EC2 compute + Postgres (us-west-2, Oregon);
  **Amazon SES** (us-west-2) sends verification/reset email from
  `no-reply@cortexeeg.org`. SES therefore processes the recipient email address.
- **Box** — encrypted off-site **backups** via `rclone`: a full `pg_dump` of the
  database **plus** admin CSV/JSON exports (`sessions.csv`, per-result JSON),
  uploaded every 6 hours, pruned after **30 days**. *These backups contain the
  full personal-data set, including emails.*
- **Google LLC** — "Sign in with Google": Google authenticates the user and
  returns an ID token; standard Google account/authentication processing applies
  on Google's side under Google's own policies.
- (Underlying compute/data accounts per the project's AWS account map; not a
  user-data processor distinction for this policy.)

## 1.4 GDPR / UK GDPR analysis

- **Roles:** the project entity is the **controller**; AWS, Box, Google act as
  **processors/independent controllers** as applicable. A controller→processor
  agreement (DPA) should be in place with each (AWS DPA, Box DPA, Google terms).
- **Legal bases (Art. 6):** (a) account + authentication + delivering the
  assessment = **performance of a contract / taking steps at the user's
  request**; (b) storing/analyzing assessment performance for the research
  purpose = **consent** (aligned with, but separate from, the IRB consent),
  with **legitimate interests** as a secondary basis for service provision; (c)
  `signup_ip` + rate limiting = **legitimate interests** (security/abuse
  prevention).
- **Special category data (Art. 9):** assessment performance reflects
  professional skill, **not** the user's own health data → Art. 9 should not be
  triggered. Confirm with counsel that nothing collected reveals the user's
  health, etc.
- **International transfer:** data is stored in the **US**; users are clinicians
  worldwide (incl. EEA/UK). Need an Art. 46 mechanism — **Standard Contractual
  Clauses** (via the AWS/Box DPAs) + a transfer-risk assessment. State this
  plainly in the policy.
- **Data-subject rights:** access, rectification, erasure, restriction,
  objection, portability, withdraw consent. ⚠️ **Currently fulfillable only
  manually** (admin CLI); see gaps.

## 1.5 US state law (CCPA/CPRA et al.) & PIPEDA

- **No "sale" or "sharing"** of personal information (no ad networks, no
  cross-context behavioral advertising). Service-provider relationships only.
- Provide CCPA rights (know/access, delete, correct, opt-out of sale/share —
  N/A) and a "we do not sell or share" statement.
- **PIPEDA (Canada):** consent-based model is satisfied; cross-border (US)
  storage must be disclosed (done in the policy).

## 1.6 Top risks / gaps to fix (priority order)

1. **No self-service export/delete endpoint (mitigated, not closed).** Per team
   decision (2026-06-18) the chosen channel is a **manual DSAR process**: users
   email mbwest@stanford.edu / ewk23@stanford.edu and the team fulfills via the
   admin CLI. The policy now states this channel. RECOMMENDED FOLLOW-UP: an
   internal runbook + response-time SLA for these requests, and a self-service
   delete endpoint later.
2. **Retention = account lifetime (per team decision 2026-06-18).** Data is kept
   while the account is active; deletion on request via the channel above. Backups
   persist up to 30 extra days. No automatic purge of inactive accounts is
   implemented — consider one later.
3. **Backups (Box) contain the full PII set**, including emails, for up to 30
   days. Erasure requests must account for backup propagation; disclose this.
4. **`signup_ip` is collected but undisclosed.** Now disclosed in the policy
   under security/technical data; confirm it's actually needed and retained no
   longer than necessary.
5. **JWT in `localStorage`** is convenient but readable by any script on the
   page (XSS exposure). Acceptable for a 6-hour functional token; keep dependency
   hygiene tight. Not a policy issue, but a security note.
6. **Self-reported `expertise` is requested but not stored** today — if Google
   signup or future code persists it, the policy already covers it; if it stays
   unstored, no action.
7. **Controller identity (resolved, pending counsel confirmation).** Stated as
   Stanford University School of Medicine (Westover Lab — CORTEX project), 453
   Quarry Rd, Office 221B, Palo Alto, CA 94304; responsible person M.B. Westover;
   legal oversight by Stanford SoM legal advisors; governing law = California.
   COUNSEL TO CONFIRM: (a) the formal GDPR controller entity name (Stanford's
   legal entity is "The Leland Stanford Junior University"); (b) whether a formal
   Art. 27 EU / UK representative is required if EEA/UK clinicians are actively
   targeted (the "Stanford legal advisors" line is not itself an in-region Art. 27
   representative).

---

# PART 2 — CORTEX Privacy Policy (public draft for `app.cortexeeg.org/privacy`)

> **DRAFT — placeholders resolved from team input on 2026-06-18; pending final
> review by Stanford legal counsel before publishing.**

**Effective date:** June 18, 2026

## Who we are

CORTEX is an online platform that helps neurologists and other EEG readers
assess and improve their skill at recognizing patterns in EEG recordings. This
Privacy Policy explains what personal data we collect when you use CORTEX at
`app.cortexeeg.org`, why we collect it, how we protect it, and the choices and
rights you have.

CORTEX is part of the Westover Lab at Stanford University School of Medicine and
is sponsored by the Clinical Data Animation Center (CDAC). The data controller is
Stanford University School of Medicine (Westover Lab — CORTEX project), 453 Quarry
Rd, Office 221B, Palo Alto, CA 94304. If you have any questions, contact us at
mbwest@stanford.edu or ewk23@stanford.edu. The individual responsible for data
protection for CORTEX is Michael B. Westover, MD, PhD, Professor of Neurology,
Stanford University School of Medicine; privacy and legal matters are overseen by
the legal advisors to Stanford University School of Medicine.

This policy covers your personal data as a **user** of CORTEX. The EEG examples
you review in the assessment are **pre-existing, de-identified research
recordings**; they are not your patients' data and are not collected from you.

> **A note on research participation.** CORTEX is used in connection with a
> research study. Separately from this Privacy Policy, you are shown a
> **research informed-consent** notice before your first assessment. That consent
> governs your participation in the research; this Privacy Policy explains how we
> handle your personal data. The Principal Investigator is Michael B. Westover,
> MD, PhD, Professor of Neurology, Stanford University School of Medicine
> (mbwest@stanford.edu).

## The data we collect

**Account information.** Your email address, the display name you choose, and an
optional self-reported professional role/expertise. If you create your account
with a password, we store a securely hashed version of your password (never the
password itself). If you use "Sign in with Google," we receive your email
address, name, and a Google account identifier from Google (see below).

**Assessment and progress data.** Your responses during assessments, whether
they were correct, how long they took, and the results we generate from them, as
well as practice/training activity and progress over time. We use this to show
you your results and to support the research purpose described above. *We
describe what this data is used for, not the proprietary methods we use to
analyze it.*

**Technical and usage data.** To keep the service secure and working, we record
limited technical information such as the IP address used at sign-up and
temporary counters used to prevent abuse (for example, limiting repeated sign-in
attempts).

**Cookies and local storage.** After you sign in, we store a temporary sign-in
token in your browser's local storage so you stay logged in for a limited time.
This is strictly functional. **We do not use advertising cookies or third-party
tracking, and we do not track you across other websites.**

## How and why we use your data

- To create and secure your account and sign you in.
- To send you account emails (such as a verification code or a password-reset
  code).
- To deliver the assessment, generate and show your results, and provide
  practice and progress features.
- To support the research purpose for which CORTEX is used.
- To protect the service against fraud, abuse, and security threats.
- To comply with our legal obligations.

## Legal bases (for users in the EEA, UK, and similar regimes)

We rely on: **performance of a contract** (to provide the account and
assessment you request); **your consent** (for participation in the research
purpose; you may withdraw it at any time); and our **legitimate interests** (to
keep the service secure and prevent abuse). Where we rely on consent, withdrawing
it will not affect the lawfulness of processing before withdrawal.

## Sign in with Google

If you choose "Continue with Google," Google handles signing you in and sends us
your email address, name, and a Google account identifier so we can create or
access your CORTEX account. We use these only for authentication and your account
profile. We do **not** receive your Google password, and we do not access your
Gmail, contacts, or other Google data. Google's handling of your information is
governed by Google's own privacy policy.

## Service providers and international transfers

We use trusted service providers to run CORTEX, who process data only on our
instructions:

- **Amazon Web Services** — hosting and database, and email delivery for account
  messages. Our servers and database are located in the **United States
  (Oregon)**.
- **Box** — encrypted, access-controlled backups.
- **Google** — "Sign in with Google" authentication, if you choose it.

Because our infrastructure is in the **United States**, if you access CORTEX from
outside the US your personal data will be transferred to and stored in the US.
Where required, we rely on appropriate safeguards for these transfers, including
**Standard Contractual Clauses** with our providers.

## How long we keep your data

We keep your account and assessment data for as long as your account is active.
You can ask us to delete your data at any time (see "Your rights and choices");
when you do, we will delete it, unless a longer period is required by law.
Short-lived items (such as email verification codes) expire within minutes.
Backups are kept for up to **30 days** and then deleted, so it can take up to 30
days for a deletion to fully propagate out of backups.

## How we protect your data

We use industry-standard safeguards, including **encryption in transit (HTTPS)**,
strong one-way hashing of passwords, protection of verification codes at rest,
time-limited sign-in tokens, access controls, rate limiting, and encrypted
off-site backups. No system is perfectly secure, but we work to protect your data
and to limit what we collect to what we need.

## Your rights and choices

Depending on where you live, you may have the right to: **access** a copy of your
data; **correct** inaccurate data; **delete** your data; **restrict** or
**object** to certain processing; receive your data in a **portable** format; and
**withdraw consent** or **withdraw from the research** at any time. We do **not
sell or share** your personal information, and we do not use it for targeted
advertising.

To exercise any of these rights — including to **access or delete** your data —
email us at mbwest@stanford.edu or ewk23@stanford.edu. We will respond within the
time required by applicable law. If you are in the EEA or UK, you also have the
right to lodge a complaint with your local data-protection authority.

## Children

CORTEX is intended for licensed clinicians and trainees and is **not directed to
anyone under 18**. We do not knowingly collect data from children.

## Changes to this policy

We may update this policy from time to time. We will post the updated version
here and change the "Effective date" above; significant changes will be
communicated as required by law.

## Contact us

Stanford University School of Medicine — Westover Lab (CORTEX project), sponsored
by the Clinical Data Animation Center (CDAC)
453 Quarry Rd, Office 221B, Palo Alto, CA 94304
mbwest@stanford.edu · ewk23@stanford.edu

This policy is governed by the laws of the State of California, United States.
