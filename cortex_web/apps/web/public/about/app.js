import { initParticleCloud } from "./scripts/demos/particle-cloud.mjs?v=20260724-spiral-position";
import { initTitleReveal } from "./scripts/demos/title-reveal.mjs?v=20260724-transition-extended-identity";

const themeToggle = document.querySelector("#theme-toggle");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const demos = [];
const titleStage = document.querySelector("#title-stage");
const graphStage = document.querySelector("#graph-stage");
const languageSelect = document.querySelector("#about-language");
const supportedLanguages = new Set([
  "en",
  "es",
  "fr",
  "de",
  "pt",
  "it",
  "zh-Hans",
  "ja",
]);

titleStage.dataset.initialization = "loading";
demos.push(initTitleReveal());
titleStage.dataset.initialization = "ready";

async function startParticleDemo() {
  graphStage.dataset.initialization = "loading";
  const demo = await initParticleCloud();
  demos.push(demo);
  graphStage.dataset.initialization = "ready";
}

graphStage.dataset.initialization = "deferred";

if (reducedMotion.matches) {
  void startParticleDemo();
} else {
  requestAnimationFrame(() => {
    window.setTimeout(() => void startParticleDemo(), 64);
  });
}

function deferDemo(stageSelector, load) {
  const stage = document.querySelector(stageSelector);
  stage.dataset.initialization = "deferred";
  let started = false;

  async function start() {
    if (started) return;
    started = true;
    stage.dataset.initialization = "loading";
    const demo = await load();
    demos.push(demo);
    stage.dataset.initialization = "ready";
  }

  if (reducedMotion.matches) {
    void start();
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    observer.disconnect();
    void start();
  }, {
    rootMargin: "150% 0px",
  });
  observer.observe(stage);
}

deferDemo("#curves-stage", async () => {
  const { initModelComparison } = await import(
    "./scripts/demos/model-comparison.mjs?v=20260724-completion-gates"
  );
  return initModelComparison();
});

deferDemo("#roc-stage", async () => {
  const { initDiscriminationComparison } = await import(
    "./scripts/demos/discrimination-comparison.mjs?v=20260724-density-labels"
  );
  return initDiscriminationComparison();
});

deferDemo("#context-stage", async () => {
  const { initContextNarrative } = await import(
    "./scripts/demos/context-narrative.mjs?v=20260724-post-fall-prose-exit"
  );
  return initContextNarrative();
});

function syncThemeToggle() {
  const isDark = document.documentElement.dataset.theme === "dark";
  const action = isDark ? "light" : "dark";
  themeToggle.setAttribute("aria-label", `Switch to ${action} theme`);
  themeToggle.setAttribute("title", `${action[0].toUpperCase()}${action.slice(1)} theme`);
}

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;

  try {
    localStorage.setItem("cortex-theme", next);
  } catch {
    // The active page still updates when storage is unavailable.
  }

  syncThemeToggle();
  for (const demo of demos) demo.redraw();
});

document.querySelectorAll(".portrait-wrap img").forEach((image) => {
  image.addEventListener("error", () => image.classList.add("is-missing"));
});

if (languageSelect) {
  try {
    const storedLanguage = localStorage.getItem("cortex-lang");
    if (supportedLanguages.has(storedLanguage)) {
      languageSelect.value = storedLanguage;
    }
  } catch {
    // The page remains usable when storage is unavailable.
  }

  languageSelect.addEventListener("change", () => {
    try {
      localStorage.setItem("cortex-lang", languageSelect.value);
    } catch {
      // The selection still updates for the active page.
    }
  });
}

syncThemeToggle();
