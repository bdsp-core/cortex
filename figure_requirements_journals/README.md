# Journal figure / display-item requirements — submission targets

Authoritative, **web-verified** figure specifications for the journals this project may submit to, so
agents stop relying on assumptions/stale memory. Compiled **2026-06-03** by four parallel research agents
(one per journal), each fetching the official publisher pages, cross-verifying across ≥2 sources, citing
exact URLs, and explicitly marking gaps. Per-journal detail + sources are in the sibling files:

- [`nejm_ai.md`](nejm_ai.md) — **primary target (Paper 2 / CORTEX)**
- [`nature_medicine.md`](nature_medicine.md)
- [`epilepsia.md`](epilepsia.md) — Paper 1 likely target (ILAE)
- [`clinical_neurophysiology.md`](clinical_neurophysiology.md)

> **Verification caveat (read this):** Several publisher author-guideline pages (NEJM AI `ai.nejm.org`,
> Nature Medicine `nature.com/nm`, Epilepsia Wiley page, Clinical Neurophysiology ScienceDirect guide)
> are **login/paywall/403-gated to automated fetch**. Where the primary page was blocked, specs were
> recovered from (a) the publisher's *public* artwork sub-pages (Nature's `research-figure-guide.nature.com`,
> Elsevier's `elsevier.com/about/policies-and-standards/...artwork...`, Wiley's `authors.wiley.com` +
> 2016 artwork PDF) and (b) corroborating search snippets. **Confidence is flagged per item in each file.**
> Re-verify the gated numbers against the live guidelines (or the editorial office) before final submission.

## Cross-journal comparison (Original/Research Article)

| Spec | NEJM AI | Nature Medicine | Epilepsia (Wiley) | Clin. Neurophysiol. (Elsevier) |
|---|---|---|---|---|
| **Display items (fig+table)** | **5** (combined) ✓verified | ~8 main + up to 10 Extended Data | **6** (combined; 4000-word art.) | Not stated for Original (Letters: 1) |
| **Main-figure format** | Vector **AI/EPS/SVG** (data); raster for photos; PDF ok | **Vector PDF/EPS only** — *no JPEG/TIFF/PNG for main figs*; ED: JPEG/TIFF/EPS | EPS/PDF (line art); TIFF/PNG (images) | EPS/PDF (vector); TIFF/JPEG (raster) |
| **Line-art DPI** | not stated by NEJM AI (flagship 1200) | prefer vector | **600–1000** | **1000** (1200 if fine lines) |
| **Halftone/photo DPI** | not stated (flagship 300) | **300** (export 450) | **300** | **300** |
| **Combination DPI** | — | — | 600 (line-art rule) | **500** |
| **Single-col width** | not stated (flagship 8.3 cm) | **89 mm** (3.5″) | ~80–85 mm (3.15″) | **90 mm** (3.54″) |
| **1.5-col width** | — | not stated | — | **140 mm** (5.51″) |
| **Double-col width** | not stated (flagship 17.4 cm) | **183 mm** (7.2″) | **180 mm** (7.09″) | **190 mm** (7.48″) |
| **Max height** | not stated | **170 mm** (6.7″) | not stated (~22.5 cm) | not stated |
| **Font family** | not stated (flagship Arial/Helv.) | **Helvetica/Arial** (sans) | Arial/Helvetica (sans) | Arial/Helvetica/Courier/Symbol/Times |
| **Body font size** | not stated (flagship 8–10 pt) | **5–7 pt** (max 7) | **≥8 pt** | **7 pt** (sub/super ≥6) |
| **Panel labels** | not stated | **8 pt bold *lowercase* a,b,c** | *lowercase* a,b,c, 12 pt | not specified |
| **Colour mode** | not stated | **RGB** | RGB online / CMYK print | **RGB** (Elsevier converts) |
| **Colour charge** | **none** | **none** | **none** | online free; **print charged** (fMRI/PET/SPECT free at editor discretion) |
| **CVD-safe palette** | not stated | **required** | recommended | recommended (5 tips) |
| **Min line weight** | not stated | 0.25–1 pt | **0.5 pt** (0.2 mm) | 0.25 pt rec / 0.1 pt min |
| **File size** | not stated | ≤50 MB (ED ≤10 MB) | <10 MB each, <500 MB total | ≤10 MB (≤7 MB if many) |
| **Generative-AI figures** | **prohibited** | **prohibited** (incl. content-aware fill) | disclose (Wiley) | prohibited for data; OK for schematics w/ disclosure |
| **Legends** | required + source line; excluded from word count; on fig page | below figure; stats details required | separate section after references | supplied **separately**, not on figure |

## DESIGN-ONCE rules (a figure that satisfies all four)

Build every paper figure to the **strictest** constraint on each axis so one source file ports across
journals with only width/label tweaks:

1. **Format → editable vector PDF** (keep the `.svg` source). PDF is accepted as a data-figure format by
   all four; Nature **rejects raster main figures**, so vector is mandatory anyway. Submit PDF; never a
   flattened raster for data figures. (Photographic/EEG-image panels: high-res raster ≥300 dpi.)
2. **Keep text live + embed fonts:** `svg.fonttype='none'`, `pdf.fonttype=42`. Do **not** outline text.
   (All four require embedded fonts; Nature/Wiley/Elsevier say embed, don't flatten.)
3. **Font = Helvetica/Arial (sans-serif).** Verified for Nature/Epilepsia/Clin-Neuro; safe for NEJM.
4. **Font size:** ⚠ **genuine conflict** — Nature caps body at **7 pt**, Epilepsia floors at **8 pt**.
   No single size satisfies both. → size to the *target* journal; for a cross-journal draft use ~7–8 pt
   and adjust. Clin-Neuro min 7 pt; sub/superscripts ≥6 pt. Never smaller than 5 pt anywhere.
5. **Panel labels = lowercase bold `a, b, c`.** ⚠ Nature **requires lowercase**; others don't forbid it.
   (Our existing figures use uppercase `A/B/C/D` — **relabel to lowercase if targeting Nature Medicine**.)
6. **Colour = RGB**, CVD-safe palette (Nature *requires* it; the NEJM 8-colour palette is CVD-aware).
   Never encode by colour alone — add position/shape/label. Wiley/Elsevier convert RGB→CMYK for print.
7. **Min line weight ≥ 0.5 pt** (Epilepsia's floor is the strictest; satisfies all).
8. **Width:** design at the target journal's exact column width (they differ by a few mm — see table).
   Safe defaults: single ≈ **89 mm/3.5″**, double ≈ **183–190 mm/7.2–7.5″**. Keep height ≤ ~170 mm.
9. **No generative-AI imagery** in any figure (prohibited by NEJM AI, Nature, Elsevier-for-data).
10. **Provenance + legends:** every number regenerated from current shipped code/data
    ([[feedback-sims-ground-in-current-code]]); legends carry n, test, CI definition; source line if reused.

These rules supersede the assumed values previously in [[reference-figure-aesthetics]] (which is now a
pointer to this directory). Our in-repo template `paper_sims/figures/fig4_validation.py` already follows
1–3, 6, 7, 9–10; the open item is **panel-label case** (uppercase now; switch to lowercase for Nature).

## Per-journal quick notes
- **NEJM AI** publishes *thin* technical specs and defers to NEJM Group's 2025 Technical Guidelines
  (editable vector for data figures; raster for photos; no SPSS/Stata exports). Hard-verified: **5**
  display items, legends+source lines, **no AI figures**, no colour charge. DPI/width/font **not stated
  by NEJM AI itself** (the 8.3/17.4 cm, Arial/Helvetica 8–10 pt, 300/1200 dpi are *flagship NEJM*, not
  confirmed for NEJM AI).
- **Nature Medicine** has the **most detailed + strictest** public spec — use it as the design ceiling.
- **Epilepsia/Clin-Neuro** follow their publishers' (Wiley/Elsevier) general artwork rules; no major
  journal-specific overrides found except Clin-Neuro's free colour for fMRI/PET/SPECT.
