// Public Terms of Service page, served at /terms (Caddy try_files → index.html;
// main.tsx renders this for the /terms path). Mirrors the Privacy Policy page
// aesthetic + i18n approach: themed CSS vars, horizontal wordmark, theme toggle,
// teal accents, sharp panel radii. All copy comes from the i18n catalogs
// ("terms.*" keys); paragraphs may carry **bold** and [label](href) markers
// rendered by renderRich. The internal draft/analysis in
// docs/TERMS_OF_SERVICE_DRAFT.md is never shipped.

import { CSSProperties, ReactNode } from "react";
import { COLORS, FONTS } from "../../ui/theme";
import { ThemeToggle } from "../theme/ThemeProvider";
import { useI18n, renderRich } from "../i18n/LanguageProvider";

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
const column: CSSProperties = { width: "100%", maxWidth: 760, margin: "0 auto" };
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

export function TermsPage() {
  const { t, lang } = useI18n();
  const r = (key: string) => renderRich(t(key));
  return (
    <div style={wrap}>
      <ThemeToggle style={{ position: "absolute", top: 24, right: 24 }} />
      <div style={column}>
        <a href="/" aria-label={t("terms.back")} style={{ display: "inline-block", marginBottom: 28 }}>
          <img
            className="cortex-logo"
            src="/cortex_logo_word_horizontal@3x.png"
            srcSet="/cortex_logo_word_horizontal@2x.png 2x, /cortex_logo_word_horizontal@3x.png 3x"
            alt={t("common.logoAlt")}
            style={{ height: 34, width: "auto", display: "block" }}
          />
        </a>

        <h1 style={titleStyle}>{t("terms.title")}</h1>
        <p style={effStyle}>{t("terms.effective")} {EFFECTIVE_DATE}</p>
        {lang !== "en" && <div style={noteStyle}>{t("privacy.translationNote")}</div>}
        <P>{r("terms.lead")}</P>

        <H2>{t("terms.acceptance.h")}</H2>
        <P>{r("terms.acceptance.body")}</P>

        <H2>{t("terms.eligibility.h")}</H2>
        <P>{r("terms.eligibility.body")}</P>

        <H2>{t("terms.service.h")}</H2>
        <P>{r("terms.service.body")}</P>

        <H2>{t("terms.research.h")}</H2>
        <div style={noteStyle}>{r("terms.research.body")}</div>

        <H2>{t("terms.accounts.h")}</H2>
        <P>{r("terms.accounts.body")}</P>

        <H2>{t("terms.use.h")}</H2>
        <P>{t("terms.use.intro")}</P>
        <ul>
          <li style={liStyle}>{r("terms.use.li1")}</li>
          <li style={liStyle}>{r("terms.use.li2")}</li>
          <li style={liStyle}>{r("terms.use.li3")}</li>
          <li style={liStyle}>{r("terms.use.li4")}</li>
          <li style={liStyle}>{r("terms.use.li5")}</li>
          <li style={liStyle}>{r("terms.use.li6")}</li>
          <li style={liStyle}>{r("terms.use.li7")}</li>
        </ul>

        <H2>{t("terms.ip.h")}</H2>
        <P>{r("terms.ip.body")}</P>

        <H2>{t("terms.medical.h")}</H2>
        <P>{r("terms.medical.body")}</P>

        <H2>{t("terms.disclaimer.h")}</H2>
        <P>{r("terms.disclaimer.body")}</P>

        <H2>{t("terms.liability.h")}</H2>
        <P>{r("terms.liability.body")}</P>

        <H2>{t("terms.termination.h")}</H2>
        <P>{r("terms.termination.body")}</P>

        <H2>{t("terms.changes.h")}</H2>
        <P>{r("terms.changes.body")}</P>

        <H2>{t("terms.law.h")}</H2>
        <P>{r("terms.law.body")}</P>

        <H2>{t("terms.privacy.h")}</H2>
        <P>{r("terms.privacy.body")}</P>

        <H2>{t("terms.contact.h")}</H2>
        <P>
          {t("terms.contact.org")}<br />
          {t("terms.contact.addr")}<br />
          {r("terms.contact.emails")}
        </P>

        <div style={{ marginTop: 40, paddingTop: 20, borderTop: `1px solid ${COLORS.borderInactive}` }}>
          <a href="/" style={linkStyle}>{t("terms.back")}</a>
        </div>
      </div>
    </div>
  );
}
