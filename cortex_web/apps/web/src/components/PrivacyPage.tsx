// Public Privacy Policy page, served at /privacy (Caddy try_files → index.html;
// main.tsx renders this instead of <App/> when the path is /privacy). Mirrors
// the auth surface aesthetic: themed CSS vars, horizontal wordmark, theme
// toggle, teal accents, sharp panel radii. All copy comes from the i18n
// catalogs (locales/*.json, "privacy.*" keys); paragraphs may carry **bold**
// and [label](href) markers rendered by renderRich. The internal
// data-protection analysis in docs/PRIVACY_POLICY_DRAFT.md is never shipped.

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

export function PrivacyPage() {
  const { t, lang } = useI18n();
  const r = (key: string) => renderRich(t(key));
  return (
    <div style={wrap}>
      <ThemeToggle style={{ position: "absolute", top: 24, right: 24 }} />
      <div style={column}>
        <a href="/" aria-label={t("privacy.back")} style={{ display: "inline-block", marginBottom: 28 }}>
          <img
            className="cortex-logo"
            src="/cortex_logo_word_horizontal@3x.png"
            srcSet="/cortex_logo_word_horizontal@2x.png 2x, /cortex_logo_word_horizontal@3x.png 3x"
            alt={t("common.logoAlt")}
            style={{ height: 34, width: "auto", display: "block" }}
          />
        </a>

        <h1 style={titleStyle}>{t("privacy.title")}</h1>
        <p style={effStyle}>{t("privacy.effective")} {EFFECTIVE_DATE}</p>
        {lang !== "en" && <div style={noteStyle}>{t("privacy.translationNote")}</div>}

        <H2>{t("privacy.whoWeAre.h")}</H2>
        <P>{r("privacy.whoWeAre.p1")}</P>
        <P>{r("privacy.whoWeAre.p2")}</P>
        <P>{r("privacy.whoWeAre.p3")}</P>
        <div style={noteStyle}>{r("privacy.whoWeAre.note")}</div>

        <H2>{t("privacy.data.h")}</H2>
        <P>{r("privacy.data.account")}</P>
        <P>{r("privacy.data.assessment")}</P>
        <P>{r("privacy.data.technical")}</P>
        <P>{r("privacy.data.cookies")}</P>

        <H2>{t("privacy.howWhy.h")}</H2>
        <ul>
          <li style={liStyle}>{t("privacy.howWhy.li1")}</li>
          <li style={liStyle}>{t("privacy.howWhy.li2")}</li>
          <li style={liStyle}>{t("privacy.howWhy.li3")}</li>
          <li style={liStyle}>{t("privacy.howWhy.li4")}</li>
          <li style={liStyle}>{t("privacy.howWhy.li5")}</li>
          <li style={liStyle}>{t("privacy.howWhy.li6")}</li>
        </ul>

        <H2>{t("privacy.legal.h")}</H2>
        <P>{r("privacy.legal.body")}</P>

        <H2>{t("privacy.google.h")}</H2>
        <P>{r("privacy.google.body")}</P>

        <H2>{t("privacy.providers.h")}</H2>
        <P>{t("privacy.providers.intro")}</P>
        <ul>
          <li style={liStyle}>{r("privacy.providers.li1")}</li>
          <li style={liStyle}>{r("privacy.providers.li2")}</li>
          <li style={liStyle}>{r("privacy.providers.li3")}</li>
        </ul>
        <P>{r("privacy.providers.body")}</P>

        <H2>{t("privacy.retention.h")}</H2>
        <P>{r("privacy.retention.body")}</P>

        <H2>{t("privacy.security.h")}</H2>
        <P>{r("privacy.security.body")}</P>

        <H2>{t("privacy.rights.h")}</H2>
        <P>{r("privacy.rights.body1")}</P>
        <P>{r("privacy.rights.body2")}</P>

        <H2>{t("privacy.children.h")}</H2>
        <P>{r("privacy.children.body")}</P>

        <H2>{t("privacy.changes.h")}</H2>
        <P>{r("privacy.changes.body")}</P>

        <H2>{t("privacy.contact.h")}</H2>
        <P>
          {t("privacy.contact.org")}<br />
          {t("privacy.contact.addr")}<br />
          {r("privacy.contact.emails")}
        </P>
        <P>{t("privacy.contact.governing")}</P>

        <div style={{ marginTop: 40, paddingTop: 20, borderTop: `1px solid ${COLORS.borderInactive}` }}>
          <a href="/" style={linkStyle}>{t("privacy.back")}</a>
        </div>
      </div>
    </div>
  );
}
