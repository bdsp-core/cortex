import {
  clamp,
  createMarkerController,
} from "../shared/chart.mjs";

const scrollSteps = 121;
const interactionRangeDvh = 220;
const completionHoldDvh = 30;
const copyTravelDvh = 22;
const particleStartDelayDvh = 125;
const wordWindows = [
  { start: -.06, end: 0 },
  { start: .015, end: .085 },
  { start: .062, end: .132 },
  { start: .109, end: .178 },
  { start: .155, end: .225 },
  { start: .202, end: .271 },
  { start: .248, end: .318 },
  { start: .295, end: .38 },
];
const underlineWindow = { start: .43, end: .51 };
const transitionWindow = { start: .57, end: .91 };
const fieldHandoffWindow = { start: .57, end: .92 };
const logoHandoffWindow = { start: .91, end: .96 };

function smoothstep(value) {
  const bounded = clamp(value);
  return bounded * bounded * (3 - 2 * bounded);
}

function windowProgress(progress, { start, end }) {
  return smoothstep((progress - start) / (end - start));
}

export function initTitleReveal() {
  const stage = document.querySelector("#title-stage");
  const track = document.querySelector("#title-scroll-track");
  const status = document.querySelector("#title-status");
  const words = [...stage.querySelectorAll("[data-title-word]")];
  const section = stage.querySelector(".title-section");
  const brandWord = stage.querySelector(".title-brand-word");
  const logoSlot = stage.querySelector(".title-logo-slot");
  const titleLogo = stage.querySelector(".title-inline-logo");
  const titleColon = stage.querySelector(".title-colon");
  const physicians = stage.querySelector(".title-physicians");
  const ledgerHeader = document.querySelector(".site-header");
  const ledgerLogo = document.querySelector(".cortex-logo img");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  stage.dataset.titleText = "CORTEX: the future of EEG training for physicians";
  stage.dataset.wordCount = String(words.length);
  stage.dataset.revealMode = "word-by-word";
  stage.dataset.underlineTarget = "physicians";
  stage.dataset.underlineStart = "after-title-complete";
  stage.dataset.logoAsset = "cortex-logo-word-horizontal";
  stage.dataset.logoDestination = "skill-bias-ledger-top-left";
  stage.dataset.handoffMode = "measured-logo-crossfade";
  stage.dataset.logoBaselineCorrection = "-0.12em";
  stage.dataset.copyTravelDvh = String(copyTravelDvh);
  stage.dataset.particleStartDelayDvh = String(particleStartDelayDvh);
  stage.dataset.completionGate = "render-before-release";
  stage.dataset.completionHoldDvh = String(completionHoldDvh);
  stage.dataset.completionState = reducedMotion.matches ? "complete" : "running";

  let targetProgress = reducedMotion.matches ? 1 : 0;
  let visibleProgress = targetProgress;
  let animationFrame = 0;
  let announcedState = "";
  let logoGeometry = {
    translateX: 0,
    translateY: 0,
    scale: 1,
  };

  function measureLogoHandoff() {
    const source = logoSlot.getBoundingClientRect();
    const target = ledgerLogo.getBoundingClientRect();
    const headerStyle = getComputedStyle(ledgerHeader);
    const targetY = Number.parseFloat(headerStyle.paddingTop);
    const scale = target.height / source.height;

    logoGeometry = {
      translateX: target.x - source.x,
      translateY: targetY - source.y,
      scale,
    };
    stage.dataset.logoSourceHeight = source.height.toFixed(2);
    stage.dataset.logoTargetHeight = target.height.toFixed(2);
    stage.dataset.logoTargetPosition = `${target.x.toFixed(2)},${targetY.toFixed(2)}`;
  }

  function render() {
    const exit = reducedMotion.matches
      ? 0
      : windowProgress(visibleProgress, transitionWindow);
    const logoMotion = reducedMotion.matches ? 0 : exit;
    const fieldHandoff = reducedMotion.matches
      ? 0
      : windowProgress(visibleProgress, fieldHandoffWindow);
    const logoHandoff = reducedMotion.matches
      ? 0
      : windowProgress(visibleProgress, logoHandoffWindow);

    const reveals = words.map((word, index) => {
      const reveal = windowProgress(visibleProgress, wordWindows[index]);
      const revealOffset = (1 - reveal) * .34;

      if (word === brandWord) {
        word.style.setProperty(
          "--title-word-offset",
          `${revealOffset.toFixed(4)}em`,
        );
        logoSlot.style.setProperty("--title-brand-reveal", reveal.toFixed(4));
        titleColon.style.setProperty(
          "--title-colon-opacity",
          (reveal * (1 - exit)).toFixed(4),
        );
        titleColon.style.setProperty(
          "--title-colon-offset",
          `${(-exit * copyTravelDvh).toFixed(4)}dvh`,
        );
      } else {
        word.style.setProperty(
          "--title-word-reveal",
          (reveal * (1 - exit)).toFixed(4),
        );
        word.style.setProperty(
          "--title-word-offset",
          `calc(${revealOffset.toFixed(4)}em - ${(exit * copyTravelDvh).toFixed(4)}dvh)`,
        );
      }
      return reveal;
    });
    const underline = windowProgress(visibleProgress, underlineWindow);
    physicians.style.setProperty("--title-underline", underline.toFixed(4));

    const currentScale = 1 + (logoGeometry.scale - 1) * logoMotion;
    titleLogo.style.setProperty(
      "--title-logo-translate-x",
      `${(logoGeometry.translateX * logoMotion).toFixed(3)}px`,
    );
    titleLogo.style.setProperty(
      "--title-logo-translate-y",
      `${(logoGeometry.translateY * logoMotion).toFixed(3)}px`,
    );
    titleLogo.style.setProperty("--title-logo-scale", currentScale.toFixed(5));
    titleLogo.style.setProperty(
      "--title-logo-opacity",
      (1 - logoHandoff).toFixed(4),
    );
    const ledgerLogoOpacity = reducedMotion.matches ? 1 : logoHandoff;
    ledgerLogo.style.setProperty(
      "--ledger-logo-opacity",
      ledgerLogoOpacity.toFixed(4),
    );
    section.style.setProperty(
      "--title-field-opacity",
      (1 - fieldHandoff).toFixed(4),
    );

    const visibleWordCount = reveals.filter((reveal) => reveal >= .995).length;
    const complete = visibleProgress >= .9995
      && underline >= .995
      && (reducedMotion.matches || logoHandoff >= .995);
    stage.dataset.titleProgress = visibleProgress.toFixed(3);
    stage.dataset.visibleWordCount = String(visibleWordCount);
    stage.dataset.underlineProgress = underline.toFixed(3);
    stage.dataset.logoMotionProgress = logoMotion.toFixed(3);
    stage.dataset.copyOpacity = (1 - exit).toFixed(3);
    stage.dataset.fieldOpacity = (1 - fieldHandoff).toFixed(3);
    stage.dataset.logoHandoffProgress = (
      reducedMotion.matches ? 1 : logoHandoff
    ).toFixed(3);
    stage.dataset.ledgerLogoOpacity = ledgerLogoOpacity.toFixed(3);
    stage.dataset.completionState = complete ? "complete" : "running";

    const nextAnnouncedState = complete
      ? "complete"
      : logoMotion > 0
        ? "handoff"
      : visibleWordCount === words.length
        ? "title-visible"
        : "revealing";
    if (nextAnnouncedState === announcedState) return;
    announcedState = nextAnnouncedState;

    if (complete) {
      status.textContent = (
        "The CORTEX logo has moved into the skill and bias demonstration ledger."
      );
    } else if (logoMotion > 0) {
      status.textContent = (
        "The title is clearing while the CORTEX logo moves to the upper-left ledger."
      );
    } else if (visibleWordCount === words.length) {
      status.textContent = (
        "CORTEX: the future of EEG training for physicians. Continue scrolling to emphasize physicians."
      );
    } else {
      status.textContent = `${visibleWordCount} of ${words.length} title words revealed.`;
    }
  }

  function animate() {
    animationFrame = 0;
    const difference = targetProgress - visibleProgress;

    if (Math.abs(difference) < .0004) {
      visibleProgress = targetProgress;
      render();
      return;
    }

    visibleProgress += difference * .065;
    render();
    animationFrame = requestAnimationFrame(animate);
  }

  function setScrollProgress(progress) {
    const bounded = clamp(progress);
    targetProgress = reducedMotion.matches ? 1 : bounded;
    stage.dataset.scrollProgress = bounded.toFixed(3);

    if (reducedMotion.matches) {
      visibleProgress = 1;
      if (animationFrame) cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      render();
      return;
    }

    if (!animationFrame) animationFrame = requestAnimationFrame(animate);
  }

  reducedMotion.addEventListener("change", () => {
    if (!reducedMotion.matches) return;
    targetProgress = 1;
    visibleProgress = 1;
    if (animationFrame) cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    render();
  });

  createMarkerController({
    stage,
    track,
    steps: scrollSteps,
    rangeDvh: interactionRangeDvh,
    reducedMotion,
    onProgress: setScrollProgress,
  });

  measureLogoHandoff();
  const resizeObserver = new ResizeObserver(() => {
    measureLogoHandoff();
    render();
  });
  resizeObserver.observe(stage.querySelector(".title-frame"));
  resizeObserver.observe(ledgerHeader);
  titleLogo.addEventListener("load", () => {
    measureLogoHandoff();
    render();
  }, { once: true });

  setScrollProgress(reducedMotion.matches ? 1 : 0);

  return {
    redraw() {
      measureLogoHandoff();
      render();
    },
  };
}
