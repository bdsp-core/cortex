// "Did you mean …?" for the signup email field. Catches near-miss typos of
// common consumer + study-institution mail domains before the account is
// created on an address that can never receive its verification code (the
// 2026-07-09 stanfodhealthcare.org incident). Server-side, /api/register
// backstops this with a DNS MX/A check; this is the friendly early hint.

const KNOWN_DOMAINS = [
  "gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com",
  "live.com", "icloud.com", "me.com", "aol.com", "proton.me", "protonmail.com",
  "comcast.net", "qq.com", "163.com",
  // real near-neighbors of the above — listed so they never get "corrected"
  "mail.com", "ymail.com",
  "stanford.edu", "stanfordhealthcare.org", "stanfordchildrens.org",
  "harvard.edu", "bidmc.harvard.edu",
];

// Real TLDs whose country/branding variants of the providers above exist
// (yahoo.ca, hotmail.ca, live.ca, proton.ch, …). A typed domain whose SLD
// exactly matches a known provider but carries one of these TLDs is a
// deliberate variant, not a typo — never "correct" it. Garbage TLDs
// (gmail.con, gmail.cmo) are absent here, so they still get suggestions.
const REAL_TLDS = new Set([
  "com", "net", "org", "edu", "gov", "mil", "int", "co", "io", "me", "ai",
  "us", "uk", "ca", "de", "fr", "it", "es", "pt", "nl", "be", "ch", "at",
  "se", "no", "dk", "fi", "ie", "au", "nz", "jp", "cn", "kr", "in", "br",
  "mx", "ru", "pl", "cz", "gr", "tr", "il", "sa", "ae", "sg", "hk", "tw",
  "th", "my", "id", "ph", "vn", "za",
]);

// Plain Levenshtein, small inputs only (domains), single-row DP.
function editDistance(a: string, b: string): number {
  const prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let diag = prev[0];
    prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cur = prev[j];
      prev[j] = Math.min(
        prev[j] + 1,                                   // deletion
        prev[j - 1] + 1,                               // insertion
        diag + (a[i - 1] === b[j - 1] ? 0 : 1),        // substitution
      );
      diag = cur;
    }
  }
  return prev[b.length];
}

/** Suggest a corrected address for a near-miss domain typo, or null.
 *  Conservative on purpose: exact list members never trigger, a domain that
 *  is still a prefix of a candidate (user mid-typing) never triggers, and
 *  short domains only tolerate distance 1. */
export function suggestEmail(email: string): string | null {
  const at = email.lastIndexOf("@");
  if (at < 1) return null;
  const local = email.slice(0, at);
  const domain = email.slice(at + 1).toLowerCase();
  if (!domain.includes(".") || domain.endsWith(".")) return null;
  if (KNOWN_DOMAINS.includes(domain)) return null;
  const dot = domain.lastIndexOf(".");
  const sld = domain.slice(0, dot);
  const tld = domain.slice(dot + 1);
  if (REAL_TLDS.has(tld)) {
    for (const cand of KNOWN_DOMAINS) {
      if (cand.slice(0, cand.lastIndexOf(".")) === sld) return null;
    }
  }
  let best: string | null = null;
  let bestD = Infinity;
  for (const cand of KNOWN_DOMAINS) {
    if (cand.startsWith(domain)) return null;  // still typing toward it
    const cap = cand.length <= 6 ? 1 : 2;
    if (Math.abs(cand.length - domain.length) > cap) continue;
    const d = editDistance(domain, cand);
    if (d > 0 && d <= cap && d < bestD) {
      best = cand;
      bestD = d;
    }
  }
  return best ? `${local}@${best}` : null;
}
