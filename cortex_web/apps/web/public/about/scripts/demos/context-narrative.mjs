import {
  clamp,
  createMarkerController,
} from "../shared/chart.mjs";

const scrollSteps = 211;
const interactionRangeDvh = 450;
const completionHoldDvh = 0;
const tokenRevealDuration = .052;
const columnWindows = [
  { start: .02, end: .285 },
  { start: .30, end: .53 },
];
const dividerWindow = { start: .015, end: .11 };
const highlightWindow = { start: .535, end: .57 };
const firstFallStart = .59;
const fallStagger = .052;
const fallDuration = .05;
const acronymOrder = ["C", "O", "R", "T", "E", "X"];
const fallWindows = acronymOrder.map((letter, index) => ({
  letter,
  start: firstFallStart + fallStagger * index,
  end: firstFallStart + fallStagger * index + fallDuration,
}));
const finalFallEnd = Number(fallWindows.at(-1).end.toFixed(3));
const textExitWindow = { start: finalFallEnd, end: .965 };
const textExitDistanceDvh = 12;
const teamEntryStart = .915;
const teamOverlapDvh = interactionRangeDvh * (1 - teamEntryStart);
const highlightHoldDvh = (
  interactionRangeDvh * (firstFallStart - highlightWindow.end)
);
const fallStaggerDvh = interactionRangeDvh * fallStagger;
const fallDurationDvh = interactionRangeDvh * fallDuration;
const textExitDurationDvh = (
  interactionRangeDvh
  * Number((textExitWindow.end - textExitWindow.start).toFixed(3))
);

function smoothstep(value) {
  const bounded = clamp(value);
  return bounded * bounded * (3 - 2 * bounded);
}

function windowProgress(progress, { start, end }) {
  return smoothstep((progress - start) / (end - start));
}

function fallingProgress(progress, { start, end }) {
  const linear = clamp((progress - start) / (end - start));
  return linear * linear;
}

function tokenizeColumn(column) {
  for (const node of [...column.childNodes]) {
    if (node.nodeType === Node.TEXT_NODE) {
      const fragment = document.createDocumentFragment();
      for (const piece of node.textContent.split(/(\s+)/)) {
        if (!piece) continue;
        if (/^\s+$/.test(piece)) {
          fragment.append(document.createTextNode(piece));
          continue;
        }

        const token = document.createElement("span");
        token.className = "context-token";
        token.dataset.contextToken = "";
        token.textContent = piece;
        fragment.append(token);
      }
      node.replaceWith(fragment);
      continue;
    }

    if (
      node.nodeType === Node.ELEMENT_NODE
      && node.matches(".context-acronym-source")
    ) {
      node.classList.add("context-token");
      node.dataset.contextToken = "";
    }
  }

  return [...column.querySelectorAll("[data-context-token]")];
}

export function initContextNarrative() {
  const stage = document.querySelector("#context-stage");
  const track = document.querySelector("#context-scroll-track");
  const section = stage.querySelector(".context-section");
  const frame = stage.querySelector(".context-frame");
  const columns = [...stage.querySelectorAll("[data-context-column]")];
  const status = document.querySelector("#context-status");
  const cloneLayer = stage.querySelector(".context-acronym-layer");
  const people = document.querySelector("#people");
  const sources = acronymOrder.map((letter) => (
    stage.querySelector(`[data-context-acronym="${letter}"]`)
  ));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const tokensByColumn = columns.map(tokenizeColumn);
  const sourceIndexes = new Map(sources.map((source, index) => [source, index]));
  const clones = sources.map((source) => {
    const clone = document.createElement("span");
    clone.className = "context-acronym-clone";
    clone.textContent = source.textContent;
    clone.dataset.contextClone = source.dataset.contextAcronym;
    cloneLayer.append(clone);
    return clone;
  });

  stage.dataset.startGate = "full-viewport";
  stage.dataset.sectionKind = "two-column-acronym-scroll-narrative";
  stage.dataset.columnCount = String(columns.length);
  stage.dataset.paragraphCount = String(
    stage.querySelectorAll(".context-column").length,
  );
  stage.dataset.headingCount = String(
    stage.querySelectorAll("h1,h2,h3,h4,h5,h6").length,
  );
  stage.dataset.technicalTermCount = "9";
  stage.dataset.revealSequence = (
    "column-one,column-two,acronym-highlight,sequential-word-fall,"
    + "paragraph-exit,team-rise,people-logo"
  );
  stage.dataset.acronym = "CORTEX";
  stage.dataset.acronymPhrase = (
    "Continuous Optimization Response Test EEG eXpertise"
  );
  stage.dataset.acronymWords = sources.map((source) => source.textContent).join(",");
  stage.dataset.institutions = "stanford,bidmc,mgh,other-emus";
  stage.dataset.logoAsset = "cortex-logo-word-horizontal";
  stage.dataset.logoDestination = "people-section-top";
  stage.dataset.acronymDestination = "below-viewport";
  stage.dataset.acronymMotion = "sequential-downward-fall";
  stage.dataset.fallOrder = sources.map((source) => source.textContent).join(",");
  stage.dataset.highlightHoldDvh = highlightHoldDvh.toFixed(1);
  stage.dataset.fallStaggerDvh = fallStaggerDvh.toFixed(1);
  stage.dataset.fallDurationDvh = fallDurationDvh.toFixed(1);
  stage.dataset.wordExitDirection = "down";
  stage.dataset.paragraphExitTrigger = "sixth-word-fall-complete";
  stage.dataset.paragraphExitDirection = "up";
  stage.dataset.paragraphExitDurationDvh = textExitDurationDvh.toFixed(1);
  stage.dataset.paragraphExitDistanceDvh = String(textExitDistanceDvh);
  stage.dataset.fastScrollGuard = "people-center-intersection";
  stage.dataset.teamHandoff = "opaque-rise";
  stage.dataset.teamOverlapDvh = teamOverlapDvh.toFixed(2);
  stage.dataset.completionGate = "render-before-release";
  stage.dataset.completionHoldDvh = String(completionHoldDvh);
  stage.dataset.completionState = reducedMotion.matches ? "complete" : "running";
  people.style.setProperty(
    "--context-team-overlap",
    `${teamOverlapDvh.toFixed(2)}dvh`,
  );

  let targetProgress = reducedMotion.matches ? 1 : 0;
  let visibleProgress = targetProgress;
  let animationFrame = 0;
  let lastMilestone = -1;
  let cloneGeometry = [];

  function measureFalls() {
    const frameBounds = frame.getBoundingClientRect();
    cloneGeometry = clones.map((clone, index) => {
      const source = sources[index];
      const sourceBounds = source.getBoundingClientRect();
      const sourceStyle = getComputedStyle(source);
      const sourceTop = sourceBounds.top - frameBounds.top;

      clone.style.left = `${(sourceBounds.left - frameBounds.left).toFixed(3)}px`;
      clone.style.top = `${sourceTop.toFixed(3)}px`;
      clone.style.fontFamily = sourceStyle.fontFamily;
      clone.style.fontSize = sourceStyle.fontSize;
      clone.style.fontWeight = sourceStyle.fontWeight;
      clone.style.letterSpacing = sourceStyle.letterSpacing;
      clone.style.lineHeight = sourceStyle.lineHeight;

      return {
        fallDistance: (
          frameBounds.height
          - sourceTop
          + sourceBounds.height
          + 56
        ),
      };
    });

    const sourceHeight = sources.reduce((sum, source) => (
      sum + source.getBoundingClientRect().height
    ), 0);
    stage.dataset.acronymSourceHeight = sourceHeight.toFixed(2);
  }

  function tokenReveal(token, columnIndex, tokenIndex) {
    const tokens = tokensByColumn[columnIndex];
    const window = columnWindows[columnIndex];
    const available = window.end - window.start - tokenRevealDuration;
    const position = tokens.length <= 1 ? 0 : tokenIndex / (tokens.length - 1);
    const start = window.start + available * position;
    return windowProgress(visibleProgress, {
      start,
      end: start + tokenRevealDuration,
    });
  }

  function render() {
    const reduced = reducedMotion.matches;
    const divider = reduced ? 1 : windowProgress(visibleProgress, dividerWindow);
    const highlight = reduced ? 1 : windowProgress(visibleProgress, highlightWindow);
    const textExit = reduced ? 0 : windowProgress(visibleProgress, textExitWindow);
    const fallProgresses = fallWindows.map((window) => (
      reduced ? 0 : fallingProgress(visibleProgress, window)
    ));
    const columnsOpacity = 1 - textExit;

    columns.forEach((column, columnIndex) => {
      tokensByColumn[columnIndex].forEach((token, tokenIndex) => {
        const reveal = reduced ? 1 : tokenReveal(token, columnIndex, tokenIndex);
        const sourceIndex = sourceIndexes.get(token);
        const sourceOpacity = (
          sourceIndex === undefined || fallProgresses[sourceIndex] <= 0
        ) ? 1 : 0;
        token.style.setProperty(
          "--context-token-opacity",
          (reveal * sourceOpacity).toFixed(4),
        );
        token.style.setProperty(
          "--context-token-offset",
          `${((1 - reveal) * 10).toFixed(2)}px`,
        );
      });
    });

    sources.forEach((source) => {
      source.style.setProperty(
        "--context-acronym-color",
        `color-mix(in srgb, var(--figure-ink) ${((1 - highlight) * 100).toFixed(1)}%, var(--context-teal))`,
      );
    });

    clones.forEach((clone, index) => {
      const geometry = cloneGeometry[index] || {
        fallDistance: frame.getBoundingClientRect().height,
      };
      const fall = fallProgresses[index];
      clone.style.opacity = fall > 0 ? "1" : "0";
      clone.style.transform = (
        `translate3d(0, ${(geometry.fallDistance * fall).toFixed(3)}px, 0)`
      );
    });

    section.style.setProperty(
      "--context-columns-opacity",
      columnsOpacity.toFixed(4),
    );
    section.style.setProperty(
      "--context-columns-offset",
      `${(-textExitDistanceDvh * textExit).toFixed(3)}dvh`,
    );
    section.style.setProperty(
      "--context-divider-opacity",
      (divider * columnsOpacity).toFixed(4),
    );
    section.style.setProperty(
      "--context-divider-scale",
      divider.toFixed(4),
    );
    const completedColumns = tokensByColumn.filter((tokens, columnIndex) => (
      tokens.every((token, tokenIndex) => (
        tokenReveal(token, columnIndex, tokenIndex) >= .995
      ))
    )).length;
    const completedFalls = fallProgresses.filter((progress) => (
      progress >= .995
    )).length;
    const activeFallIndex = fallProgresses.findIndex((progress) => (
      progress > .001 && progress < .995
    ));
    const complete = reduced || visibleProgress >= .9995;

    stage.dataset.narrativeProgress = visibleProgress.toFixed(3);
    stage.dataset.visibleColumnCount = String(reduced ? 2 : completedColumns);
    stage.dataset.highlightProgress = highlight.toFixed(3);
    stage.dataset.fallSequenceProgress = (
      reduced
        ? 0
        : fallProgresses.reduce((sum, progress) => sum + progress, 0)
          / fallProgresses.length
    ).toFixed(3);
    stage.dataset.fallenWordCount = String(reduced ? 0 : completedFalls);
    stage.dataset.activeFallWord = (
      activeFallIndex >= 0 ? sources[activeFallIndex].textContent : "none"
    );
    stage.dataset.teamHandoffProgress = (
      reduced
        ? 1
        : windowProgress(visibleProgress, { start: teamEntryStart, end: 1 })
    ).toFixed(3);
    stage.dataset.columnsOpacity = columnsOpacity.toFixed(3);
    stage.dataset.paragraphExitProgress = textExit.toFixed(3);
    stage.dataset.completionState = complete ? "complete" : "running";

    const milestone = complete
      ? 7
      : visibleProgress >= teamEntryStart
        ? 6
        : textExit > .01
          ? 5
          : completedFalls > 0 || activeFallIndex >= 0
            ? 4
            : highlight > .01
              ? 3
              : completedColumns;
    if (milestone === lastMilestone) return;
    lastMilestone = milestone;

    const messages = [
      "The two-column CORTEX narrative is ready.",
      "The first narrative column is visible.",
      "Both narrative columns are visible.",
      "The six words that form CORTEX are highlighted.",
      "The highlighted CORTEX words are falling in sequence.",
      "All six words have fallen. The narrative is moving upward.",
      "The team section is rising from below.",
      "The CORTEX word fall is complete. Meet the team follows.",
    ];
    status.textContent = messages[milestone];
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

  const releaseObserver = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) setScrollProgress(1);
  }, {
    rootMargin: "-45% 0px -45% 0px",
  });
  releaseObserver.observe(people);

  const resizeObserver = new ResizeObserver(() => {
    measureFalls();
    render();
  });
  resizeObserver.observe(frame);
  resizeObserver.observe(stage.querySelector(".context-columns"));

  measureFalls();
  setScrollProgress(reducedMotion.matches ? 1 : 0);

  return {
    redraw() {
      measureFalls();
      render();
    },
  };
}
