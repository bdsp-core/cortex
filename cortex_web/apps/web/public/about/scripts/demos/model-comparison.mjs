import { methods, stoppingCurves } from "../data/stopping-results.mjs";
import {
  readerQuestionCounts,
  readerQuestionMaximum,
  readersPerMethod,
} from "../data/reader-distributions.mjs";
import {
  clamp,
  createMarkerController,
  cssVar,
  lerp,
  observeCanvas,
  scale,
} from "../shared/chart.mjs";

const scrollSteps = 201;
const completionHoldDvh = 30;
const stoppingPanel = {
  title: "Stopping trajectories",
  xLabel: "Questions administered",
  yLabel: "Readers still testing (%)",
  xDomain: [0, 900],
  yDomain: [0, 1.02],
  xTicks: [0, 300, 600, 900],
  yTicks: [0, .25, .5, .75, 1],
  yFormat: (value) => String(Math.round(value * 100)),
  curves: stoppingCurves,
};

function fraction(value) {
  return value - Math.floor(value);
}

const distributions = methods.map((method, methodIndex) => ({
  ...method,
  points: readerQuestionCounts[method.key]
    .map((questions, readerIndex) => ({
      normalized: questions / readerQuestionMaximum,
      questions,
      // Fixed jitter keeps all readers legible without introducing runtime randomness.
      jitter: .17 + fraction(
        Math.sin((readerIndex + 1) * 12.9898 + (methodIndex + 1) * 78.233)
          * 43758.5453,
      ) * .66,
    }))
    .sort((first, second) => first.normalized - second.normalized),
}));

function rgba(hexColor, alpha) {
  const normalized = hexColor.replace("#", "");
  const red = Number.parseInt(normalized.slice(0, 2), 16);
  const green = Number.parseInt(normalized.slice(2, 4), 16);
  const blue = Number.parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function quantile(sortedValues, proportion) {
  if (!sortedValues.length) return 0;
  const position = (sortedValues.length - 1) * proportion;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  return lerp(
    sortedValues[lowerIndex],
    sortedValues[upperIndex],
    position - lowerIndex,
  );
}

function revealedPoints(data, threshold) {
  if (!data.length || threshold < data[0][0]) return [];
  const visible = data.filter((point) => point[0] <= threshold).map((point) => [...point]);
  const next = data.find((point) => point[0] > threshold);
  const previous = visible.at(-1);

  if (previous && next && threshold > previous[0]) {
    const progress = (threshold - previous[0]) / (next[0] - previous[0]);
    visible.push([
      threshold,
      lerp(previous[1], next[1], progress),
      lerp(previous[2], next[2], progress),
      lerp(previous[3], next[3], progress),
    ]);
  }

  return visible;
}

export function initModelComparison() {
  const canvas = document.querySelector("#curves-canvas");
  const context = canvas.getContext("2d");
  const stage = document.querySelector("#curves-stage");
  const status = document.querySelector("#curves-status");
  const counter = document.querySelector("#evidence-counter");
  const count = document.querySelector("#evidence-count");
  const track = document.querySelector("#curves-scroll-track");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let targetProgress = reducedMotion.matches ? 1 : 0;
  let visibleProgress = targetProgress;
  let animationFrame = 0;
  let lastMilestone = -1;
  let currentEvidence = -1;

  stage.dataset.startGate = "full-viewport";
  stage.dataset.curveProgress = visibleProgress.toFixed(3);
  stage.dataset.readerCountPerMethod = String(readersPerMethod);
  stage.dataset.distributionCount = String(distributions.length);
  stage.dataset.rightPanel = "individual-reader-stopping-counts";
  stage.dataset.questionHorizon = String(readerQuestionMaximum);
  stage.dataset.randomQuotaEndpoint = "0";
  stage.dataset.randomQuotaEndpointQuestion = "588";
  stage.dataset.completionGate = "render-before-release";
  stage.dataset.completionHoldDvh = String(completionHoldDvh);
  stage.dataset.completionState = reducedMotion.matches ? "complete" : "running";

  function layout() {
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const isNarrow = width < 760;

    if (isNarrow) {
      const top = Math.max(172, height * .215);
      const side = Math.max(56, width * .145);
      const bottom = 42;
      const gap = Math.max(78, height * .095);
      const panelHeight = (height - top - bottom - gap) / 2;
      return {
        width,
        height,
        isNarrow,
        plots: [
          { left: side, top, width: width - side - 18, height: panelHeight },
          {
            left: side + 22,
            top: top + panelHeight + gap,
            width: width - side - 42,
            height: panelHeight,
          },
        ],
      };
    }

    const side = Math.max(68, width * .05);
    const gap = Math.max(90, width * .065);
    const top = Math.max(160, height * .2);
    const bottom = Math.max(70, height * .075);
    const panelWidth = (width - side * 2 - gap) / 2;
    return {
      width,
      height,
      isNarrow,
      plots: [
        { left: side, top, width: panelWidth, height: height - top - bottom },
        {
          left: side + panelWidth + gap + 26,
          top,
          width: panelWidth - 26,
          height: height - top - bottom,
        },
      ],
    };
  }

  function pointToCanvas(point, panel, plot) {
    return {
      x: scale(point[0], panel.xDomain, [plot.left, plot.left + plot.width]),
      y: scale(point[1], panel.yDomain, [plot.top + plot.height, plot.top]),
    };
  }

  function drawStoppingAxes(plot, isNarrow) {
    const ink = cssVar("--figure-ink");
    const grid = cssVar("--figure-grid");
    const serif = cssVar("--research-serif");

    context.save();
    context.strokeStyle = grid;
    context.lineWidth = 1;
    context.setLineDash([7, 6]);

    for (const tick of stoppingPanel.xTicks) {
      const x = scale(tick, stoppingPanel.xDomain, [plot.left, plot.left + plot.width]);
      context.beginPath();
      context.moveTo(x, plot.top);
      context.lineTo(x, plot.top + plot.height);
      context.stroke();
    }

    for (const tick of stoppingPanel.yTicks) {
      const y = scale(tick, stoppingPanel.yDomain, [plot.top + plot.height, plot.top]);
      context.beginPath();
      context.moveTo(plot.left, y);
      context.lineTo(plot.left + plot.width, y);
      context.stroke();
    }

    context.setLineDash([]);
    context.strokeStyle = ink;
    context.lineWidth = 1.25;
    context.strokeRect(plot.left, plot.top, plot.width, plot.height);

    context.fillStyle = ink;
    context.font = `${isNarrow ? 10 : 12}px ${serif}`;
    context.textAlign = "center";
    context.textBaseline = "top";
    for (const tick of stoppingPanel.xTicks) {
      const x = scale(tick, stoppingPanel.xDomain, [plot.left, plot.left + plot.width]);
      context.fillText(String(tick), x, plot.top + plot.height + 8);
    }

    context.textAlign = "right";
    context.textBaseline = "middle";
    for (const tick of stoppingPanel.yTicks) {
      const y = scale(tick, stoppingPanel.yDomain, [plot.top + plot.height, plot.top]);
      context.fillText(stoppingPanel.yFormat(tick), plot.left - 9, y);
    }

    context.font = `700 ${isNarrow ? 13 : 16}px ${serif}`;
    context.textAlign = "center";
    context.textBaseline = "bottom";
    context.fillText(
      stoppingPanel.title,
      plot.left + plot.width / 2,
      plot.top - (isNarrow ? 14 : 18),
    );

    context.font = `700 ${isNarrow ? 11 : 14}px ${serif}`;
    context.textBaseline = "top";
    context.fillText(
      stoppingPanel.xLabel,
      plot.left + plot.width / 2,
      plot.top + plot.height + (isNarrow ? 22 : 27),
    );

    context.save();
    context.translate(
      plot.left - (isNarrow ? 34 : 44),
      plot.top + plot.height / 2,
    );
    context.rotate(-Math.PI / 2);
    context.textBaseline = "middle";
    context.fillText(stoppingPanel.yLabel, 0, 0);
    context.restore();
    context.restore();
  }

  function drawMarker(point, method, plot, size) {
    const { x, y } = pointToCanvas(point, stoppingPanel, plot);
    context.beginPath();

    if (method.marker === "triangle") {
      context.moveTo(x, y - size);
      context.lineTo(x + size * .9, y + size * .75);
      context.lineTo(x - size * .9, y + size * .75);
      context.closePath();
    } else if (method.marker === "square") {
      context.rect(x - size * .75, y - size * .75, size * 1.5, size * 1.5);
    } else {
      context.arc(x, y, size * .78, 0, Math.PI * 2);
    }

    context.fill();
  }

  function drawStoppingCurve(plot, method, progress, isNarrow) {
    const source = stoppingPanel.curves[method.key];
    const threshold = progress * stoppingPanel.xDomain[1];
    const points = revealedPoints(source, threshold);
    if (!points.length) return;

    const color = cssVar(method.color);
    if (points.length > 1) {
      context.fillStyle = rgba(color, .14);
      context.beginPath();
      points.forEach((point, index) => {
        const mapped = pointToCanvas([point[0], point[3]], stoppingPanel, plot);
        if (index === 0) context.moveTo(mapped.x, mapped.y);
        else context.lineTo(mapped.x, mapped.y);
      });
      [...points].reverse().forEach((point) => {
        const mapped = pointToCanvas([point[0], point[2]], stoppingPanel, plot);
        context.lineTo(mapped.x, mapped.y);
      });
      context.closePath();
      context.fill();
    }

    context.strokeStyle = color;
    context.lineWidth = isNarrow ? 1.8 : 2.3;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.setLineDash(method.dash);
    context.beginPath();
    points.forEach((point, index) => {
      const mapped = pointToCanvas(point, stoppingPanel, plot);
      if (index === 0) context.moveTo(mapped.x, mapped.y);
      else context.lineTo(mapped.x, mapped.y);
    });
    context.stroke();
    context.setLineDash([]);

    context.fillStyle = color;
    const markerSize = isNarrow ? 3 : 4.4;
    for (const point of source) {
      if (point[0] > threshold) break;
      drawMarker(point, method, plot, markerSize);
    }
  }

  function drawDistributionPanel(plot, progress, isNarrow) {
    const ink = cssVar("--figure-ink");
    const grid = cssVar("--figure-grid");
    const serif = cssVar("--research-serif");
    const stripGap = isNarrow ? 8 : 12;
    const stripHeight = (plot.height - stripGap * 4) / 5;
    const ticks = [0, .2, .4, .6, .8, 1];
    const revealThreshold = progress + .0001;

    context.save();
    context.fillStyle = ink;
    context.font = `700 ${isNarrow ? 13 : 16}px ${serif}`;
    context.textAlign = "center";
    context.textBaseline = "bottom";
    context.fillText(
      "Individual-reader stopping counts",
      plot.left + plot.width / 2,
      plot.top - (isNarrow ? 14 : 18),
    );

    distributions.forEach((distribution, rowIndex) => {
      const rowTop = plot.top + rowIndex * (stripHeight + stripGap);
      const color = cssVar(distribution.color);

      context.strokeStyle = grid;
      context.lineWidth = 1;
      context.setLineDash([7, 6]);
      for (const tick of ticks) {
        const x = scale(tick, [0, 1], [plot.left, plot.left + plot.width]);
        context.beginPath();
        context.moveTo(x, rowTop);
        context.lineTo(x, rowTop + stripHeight);
        context.stroke();
      }

      context.setLineDash([]);
      context.strokeStyle = ink;
      context.lineWidth = 1.1;
      context.strokeRect(plot.left, rowTop, plot.width, stripHeight);

      context.fillStyle = ink;
      context.font = `700 ${isNarrow ? 9 : 11}px ${serif}`;
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(
        distribution.label,
        plot.left - (isNarrow ? 7 : 11),
        rowTop + stripHeight / 2,
      );

      const visible = distribution.points.filter(
        (point) => point.normalized <= revealThreshold,
      );

      context.save();
      context.beginPath();
      context.rect(plot.left, rowTop, plot.width, stripHeight);
      context.clip();
      context.fillStyle = color;
      for (const point of visible) {
        const fade = progress >= .999
          ? 1
          : clamp((progress - point.normalized + .025) / .05);
        if (fade <= 0) continue;
        const x = scale(point.normalized, [0, 1], [plot.left, plot.left + plot.width]);
        const y = rowTop + point.jitter * stripHeight;
        context.globalAlpha = fade * .42;
        context.beginPath();
        context.arc(x, y, isNarrow ? 1.65 : 2.25, 0, Math.PI * 2);
        context.fill();
      }
      context.globalAlpha = 1;

      if (visible.length >= 3) {
        const values = visible.map((point) => point.normalized);
        const lower = quantile(values, .25);
        const median = quantile(values, .5);
        const upper = quantile(values, .75);
        const medianX = scale(median, [0, 1], [plot.left, plot.left + plot.width]);

        context.strokeStyle = ink;
        context.lineWidth = 1.25;
        context.setLineDash([4, 3]);
        context.beginPath();
        context.moveTo(medianX, rowTop + 3);
        context.lineTo(medianX, rowTop + stripHeight - 3);
        context.stroke();
        context.setLineDash([]);

        context.fillStyle = ink;
        context.globalAlpha = .86;
        context.font = `${isNarrow ? 8 : 9.5}px ${serif}`;
        context.textAlign = "right";
        context.textBaseline = "top";
        context.fillText(
          `median ${median.toFixed(2)} | IQR ${lower.toFixed(2)} to ${upper.toFixed(2)}`,
          plot.left + plot.width - 5,
          rowTop + 4,
        );
        context.globalAlpha = 1;
      }
      context.restore();

      if (rowIndex !== distributions.length - 1) return;

      context.fillStyle = ink;
      context.font = `${isNarrow ? 9 : 11}px ${serif}`;
      context.textAlign = "center";
      context.textBaseline = "top";
      for (const tick of ticks) {
        const x = scale(tick, [0, 1], [plot.left, plot.left + plot.width]);
        context.fillText(tick.toFixed(1), x, rowTop + stripHeight + 7);
      }
      context.font = `700 ${isNarrow ? 11 : 14}px ${serif}`;
      context.fillText(
        "Normalized question count",
        plot.left + plot.width / 2,
        rowTop + stripHeight + (isNarrow ? 21 : 25),
      );
    });
    context.restore();
  }

  function draw() {
    const graph = layout();
    context.clearRect(0, 0, graph.width, graph.height);

    const stoppingPlot = graph.plots[0];
    drawStoppingAxes(stoppingPlot, graph.isNarrow);
    context.save();
    context.beginPath();
    context.rect(
      stoppingPlot.left,
      stoppingPlot.top,
      stoppingPlot.width,
      stoppingPlot.height,
    );
    context.clip();
    for (const method of methods) {
      drawStoppingCurve(stoppingPlot, method, visibleProgress, graph.isNarrow);
    }
    context.restore();

    drawDistributionPanel(graph.plots[1], visibleProgress, graph.isNarrow);
  }

  function syncCounter() {
    const nextEvidence = Math.round(visibleProgress * readerQuestionMaximum);
    if (nextEvidence === currentEvidence) return;
    currentEvidence = nextEvidence;
    count.textContent = String(currentEvidence);
    counter.setAttribute(
      "aria-label",
      `${currentEvidence} of ${readerQuestionMaximum} questions in the stopping horizon`,
    );
  }

  function syncCompletionState() {
    const isComplete = currentEvidence === readerQuestionMaximum
      && visibleProgress >= .9995;
    stage.dataset.completionState = isComplete ? "complete" : "running";
  }

  function announceProgress() {
    const milestone = Math.round(visibleProgress * 4);
    if (milestone === lastMilestone) return;
    lastMilestone = milestone;

    if (milestone === 0) {
      status.textContent = "The stopping curves and five reader distributions are ready.";
    } else if (milestone === 4) {
      status.textContent = "All stopping trajectories and 1,250 reader outcomes are fully revealed.";
    } else {
      status.textContent = `The stopping trajectories and reader distributions are ${milestone * 25} percent revealed.`;
    }
  }

  function animate() {
    animationFrame = 0;
    const difference = targetProgress - visibleProgress;

    if (Math.abs(difference) < .0004) {
      visibleProgress = targetProgress;
      stage.dataset.curveProgress = visibleProgress.toFixed(3);
      syncCounter();
      syncCompletionState();
      announceProgress();
      draw();
      return;
    }

    visibleProgress += difference * .105;
    stage.dataset.curveProgress = visibleProgress.toFixed(3);
    syncCounter();
    syncCompletionState();
    announceProgress();
    draw();
    animationFrame = requestAnimationFrame(animate);
  }

  function setScrollProgress(progress) {
    const bounded = clamp(progress);
    targetProgress = reducedMotion.matches ? 1 : bounded;
    stage.dataset.scrollProgress = bounded.toFixed(3);

    if (bounded >= .999) {
      targetProgress = 1;
      visibleProgress = 1;
      if (animationFrame) cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      stage.dataset.curveProgress = "1.000";
      syncCounter();
      syncCompletionState();
      announceProgress();
      draw();
      return;
    }

    syncCompletionState();
    if (!animationFrame) animationFrame = requestAnimationFrame(animate);
  }

  reducedMotion.addEventListener("change", () => {
    if (!reducedMotion.matches) return;
    targetProgress = 1;
    if (!animationFrame) animationFrame = requestAnimationFrame(animate);
  });

  createMarkerController({
    stage,
    track,
    steps: scrollSteps,
    rangeDvh: 150,
    reducedMotion,
    onProgress: setScrollProgress,
  });

  const canvasObserver = observeCanvas(canvas, context, draw);
  syncCounter();
  setScrollProgress(reducedMotion.matches ? 1 : 0);

  return {
    redraw: canvasObserver.redraw,
  };
}
