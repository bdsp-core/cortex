// App-wide language switching. Mirrors ThemeProvider: the active language is
// React state, persisted to localStorage["cortex-lang"] and reflected onto
// <html lang>. Catalogs are flat key -> string maps (see locales/*.json);
// lookups fall back to English, then to the key itself, so a missing
// translation never blanks the UI. Scope today: auth screens + footer + the
// /privacy page. Adding a language = drop in one JSON file + one LANGS entry.
//
// "CORTEX" and other proper nouns are never keys — they stay literal inside the
// translated strings.

import {
  createContext, useCallback, useContext, useEffect, useState, ReactNode,
} from "react";
import en from "./locales/en.json";
import es from "./locales/es.json";
import fr from "./locales/fr.json";
import de from "./locales/de.json";
import pt from "./locales/pt.json";
import it from "./locales/it.json";
import zhHans from "./locales/zh-Hans.json";
import ja from "./locales/ja.json";

export type Lang = "en" | "es" | "fr" | "de" | "pt" | "it" | "zh-Hans" | "ja";

// Autonyms shown in the selector — never translated.
export const LANGS: { code: Lang; label: string }[] = [
  { code: "en", label: "English" },
  { code: "es", label: "Español" },
  { code: "fr", label: "Français" },
  { code: "de", label: "Deutsch" },
  { code: "pt", label: "Português" },
  { code: "it", label: "Italiano" },
  { code: "zh-Hans", label: "中文" },
  { code: "ja", label: "日本語" },
];

type Catalog = Record<string, string>;
const CATALOGS = {
  en, es, fr, de, pt, it, "zh-Hans": zhHans, ja,
} as Record<Lang, Catalog>;

const STORAGE_KEY = "cortex-lang";

function isLang(s: string | null): s is Lang {
  return !!s && Object.prototype.hasOwnProperty.call(CATALOGS, s);
}

function initialLang(): Lang {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (isLang(s)) return s;
  } catch { /* blocked storage */ }
  return "en";
}

function interpolate(s: string, vars?: Record<string, string | number>): string {
  if (!vars) return s;
  return s.replace(/\{(\w+)\}/g, (_, k) => (k in vars ? String(vars[k]) : `{${k}}`));
}

export type TFn = (key: string, vars?: Record<string, string | number>) => string;

interface I18nCtx {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: TFn;
}

const Ctx = createContext<I18nCtx | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);

  const setLang = useCallback((l: Lang) => {
    try { localStorage.setItem(STORAGE_KEY, l); } catch { /* blocked storage */ }
    setLangState(l);
  }, []);

  useEffect(() => {
    if (typeof document !== "undefined") document.documentElement.lang = lang;
  }, [lang]);

  const t = useCallback<TFn>((key, vars) => {
    const s = CATALOGS[lang][key] ?? CATALOGS.en[key] ?? key;
    return interpolate(s, vars);
  }, [lang]);

  return <Ctx.Provider value={{ lang, setLang, t }}>{children}</Ctx.Provider>;
}

export function useI18n(): I18nCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useI18n must be used within LanguageProvider");
  return c;
}

// Render a markdown-lite string: **bold** -> <strong>, [label](href) -> <a>.
// Lets catalog strings carry the same inline emphasis/links the source JSX had,
// while staying a single translatable unit (translators keep the markers).
const RICH_RE = /\*\*(.+?)\*\*|\[([^\]]+)\]\(([^)]+)\)/g;
export function renderRich(s: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  RICH_RE.lastIndex = 0;
  while ((m = RICH_RE.exec(s)) !== null) {
    if (m.index > last) out.push(s.slice(last, m.index));
    if (m[1] !== undefined) {
      out.push(<strong key={key++}>{m[1]}</strong>);
    } else {
      out.push(
        <a key={key++} href={m[3]}
          style={{ color: "var(--teal-deep)", fontWeight: 600, textDecoration: "none" }}>
          {m[2]}
        </a>,
      );
    }
    last = RICH_RE.lastIndex;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}
