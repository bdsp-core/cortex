// Public citation / attribution page at /citation (Caddy try_files → index.html;
// main.tsx renders this for the /citation path). Mirrors the Privacy / Terms
// page aesthetic (themed CSS vars, horizontal wordmark, theme toggle, teal
// accents, sharp panel radii). English-only: the content is citation data
// (BibTeX, author names, the formatted reference) which is not translated.
// Functional-level only — no proprietary engine internals.

import { CSSProperties, ReactNode } from "react";
import { COLORS, FONTS } from "../../ui/theme";
import { ThemeToggle } from "../theme/ThemeProvider";

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
  fontFamily: FONTS.sans, fontSize: 30, fontWeight: 700, letterSpacing: "-0.01em",
  color: COLORS.textPrimary, margin: "0 0 4px",
};
const ledeStyle: CSSProperties = { fontSize: 13, color: COLORS.textFaint, margin: "0 0 8px" };
const h2Style: CSSProperties = {
  fontFamily: FONTS.sans, fontSize: 18, fontWeight: 600, color: COLORS.textPrimary,
  margin: "32px 0 10px", paddingTop: 20, borderTop: `1px solid ${COLORS.borderInactive}`,
};
const pStyle: CSSProperties = { fontSize: 15, lineHeight: 1.7, color: COLORS.textBody, margin: "0 0 14px" };
const liStyle: CSSProperties = { fontSize: 15, lineHeight: 1.7, color: COLORS.textBody, marginBottom: 6 };
const linkStyle: CSSProperties = { color: "var(--teal-deep)", fontWeight: 600, textDecoration: "none" };
const noteStyle: CSSProperties = {
  fontSize: 14, lineHeight: 1.7, color: COLORS.textBody, background: COLORS.cardAlt,
  borderLeft: "3px solid var(--teal)", borderRadius: "var(--radius-ctl)",
  padding: "14px 16px", margin: "0 0 14px",
};
const codeStyle: CSSProperties = {
  display: "block", whiteSpace: "pre", overflowX: "auto",
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  fontSize: 12.5, lineHeight: 1.6, color: COLORS.textPrimary,
  background: COLORS.cardAlt, border: `1px solid ${COLORS.borderInactive2}`,
  borderRadius: "var(--radius-ctl)", padding: "14px 16px", margin: "0 0 16px",
};

function H2({ children }: { children: ReactNode }) { return <h2 style={h2Style}>{children}</h2>; }
function P({ children }: { children: ReactNode }) { return <p style={pStyle}>{children}</p>; }
function Mail({ a }: { a: string }) { return <a style={linkStyle} href={`mailto:${a}`}>{a}</a>; }
function Ext({ href }: { href: string }) {
  return <a style={linkStyle} href={href}>{href.replace(/^https?:\/\//, "")}</a>;
}

const BIBTEX = `@software{cortex2026,
  author    = {Keldsen, Elijah W. and Westover, M. Brandon},
  title     = {CORTEX: An Adaptive Platform for EEG Interpretation
               Skill Assessment and Training},
  year      = {2026},
  publisher = {Westover Lab, Stanford University School of Medicine},
  url       = {https://app.cortexeeg.org}
}

% Peer-reviewed methodology paper (forthcoming) — the full reference and DOI
% will be added here on publication.`;

export function CitationPage() {
  return (
    <div style={wrap}>
      <ThemeToggle style={{ position: "absolute", top: 24, right: 24 }} />
      <div style={column}>
        <a href="/" aria-label="Back to CORTEX" style={{ display: "inline-block", marginBottom: 28 }}>
          <img
            className="cortex-logo"
            src="/cortex_logo_word_horizontal@3x.png"
            srcSet="/cortex_logo_word_horizontal@2x.png 2x, /cortex_logo_word_horizontal@3x.png 3x"
            alt="CORTEX EEG Skill Certification"
            style={{ height: 34, width: "auto", display: "block" }}
          />
        </a>

        <h1 style={titleStyle}>How to cite CORTEX</h1>
        <p style={ledeStyle}>Attribution requirements for using the CORTEX methodology</p>

        <H2>Attribution policy</H2>
        <P>
          The CORTEX platform and the adaptive testing-and-learning methodology implemented in its
          engine are the scholarly work of the Westover Lab at Stanford University School of Medicine,
          sponsored by the Clinical Data Animation Center (CDAC).
        </P>
        <P>
          We share this methodology openly for research, education, and other non-commercial use. If you
          implement, reimplement, adapt, extend, or take direct inspiration from the CORTEX methodology,
          whether for EEG or any other assessment or learning domain, you must give clear, visible
          attribution by citing the works below.
        </P>
        <div style={noteStyle}>
          Commercial or monetized use (any product, service, or offering that is sold, licensed, or
          otherwise generates revenue using the CORTEX methodology) requires prior written permission.
          Please contact us before any such use (see "Commercial and product use" below).
        </div>

        <H2>When citation is required</H2>
        <ul>
          <li style={liStyle}>You use the CORTEX software or any part of its code.</li>
          <li style={liStyle}>You reimplement the CORTEX methodology from our publications or documentation.</li>
          <li style={liStyle}>You adapt or generalize the methodology to another assessment or learning domain.</li>
          <li style={liStyle}>You use the CORTEX approach as the basis or inspiration for your own method.</li>
        </ul>

        <H2>How to cite</H2>
        <P>Please cite both the methodology and the software.</P>
        <P>
          The peer-reviewed methodology paper is <strong>forthcoming</strong>. This page will be updated
          with the full reference and DOI on publication; until then, please cite the software and
          project below and link to <Ext href="https://app.cortexeeg.org" />.
        </P>
        <P>
          <strong>Preferred citation (software).</strong> Keldsen EW, Westover MB. CORTEX: an adaptive
          platform for EEG interpretation skill assessment and training. Westover Lab, Stanford
          University School of Medicine; 2026. https://app.cortexeeg.org
        </P>

        <H2>BibTeX</H2>
        <pre style={codeStyle}>{BIBTEX}</pre>

        <H2>In-text attribution</H2>
        <div style={noteStyle}>
          This work uses (or adapts) the CORTEX adaptive testing-and-learning methodology developed by
          the Westover Lab at Stanford University School of Medicine (Keldsen &amp; Westover, 2026;
          https://app.cortexeeg.org).
        </div>

        <H2>Commercial and product use</H2>
        <P>
          To use the CORTEX methodology in a commercial product or paid service, you must obtain prior
          written permission. Please contact <Mail a="mbwest@stanford.edu" /> and{" "}
          <Mail a="ewk23@stanford.edu" /> (and, where applicable, Stanford's Office of Technology
          Licensing) before any commercial or monetized use.
        </P>

        <H2>Questions</H2>
        <P>
          Questions about citing or using CORTEX? Contact <Mail a="mbwest@stanford.edu" /> or{" "}
          <Mail a="ewk23@stanford.edu" />.
        </P>

        <div style={{ marginTop: 40, paddingTop: 20, borderTop: `1px solid ${COLORS.borderInactive}` }}>
          <a href="/" style={linkStyle}>Back to CORTEX</a>
        </div>
      </div>
    </div>
  );
}
