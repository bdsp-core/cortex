import {
  cortexQuestionMaximum,
  discriminationCohorts,
  discriminationTrajectories,
  domainDiscriminationTrajectories,
  experiencedMeanStopRatio,
  readersPerDomain,
} from "../data/discrimination-trajectories.mjs";
import {
  clamp,
  createMarkerController,
  cssVar,
  lerp,
  observeCanvas,
} from "../shared/chart.mjs";

const scrollSteps = cortexQuestionMaximum + 1;
const completionHoldDvh = 30;
const innerQuestionArcs = Array.from(
  { length: Math.floor((cortexQuestionMaximum - 1) / 100) },
  (_, index) => (index + 1) * 100,
);
const angleTicks = [
  { angle: 0, auroc: "0.50" },
  { angle: Math.PI / 4, auroc: "0.625" },
  { angle: Math.PI / 2, auroc: "0.75" },
  { angle: 3 * Math.PI / 4, auroc: "0.875" },
  { angle: Math.PI, auroc: "1.00" },
];
const densityAxisTitleDomains = new Set(["spike", "lrda"]);

function aurocToAngle(auroc) {
  return clamp((auroc - .5) / .5) * Math.PI;
}

function polarPoint(center, radius, angle) {
  return {
    x: center.x + Math.cos(angle) * radius,
    y: center.y - Math.sin(angle) * radius,
  };
}

function revealedTrajectoryPoints(points, threshold) {
  const bounded = clamp(threshold, 0, points.at(-1).question);
  const lastIndex = Math.floor(bounded);
  const visible = points.slice(0, lastIndex + 1);

  if (lastIndex >= points.length - 1 || bounded === lastIndex) {
    return visible;
  }

  const current = points[lastIndex];
  const next = points[lastIndex + 1];
  const progress = bounded - lastIndex;
  visible.push({
    question: bounded,
    auroc: lerp(current.auroc, next.auroc, progress),
  });
  return visible;
}

function drawSemicirclePath(context, center, radius) {
  const samples = 96;
  context.beginPath();
  for (let index = 0; index <= samples; index += 1) {
    const point = polarPoint(center, radius, Math.PI * index / samples);
    if (index === 0) context.moveTo(point.x, point.y);
    else context.lineTo(point.x, point.y);
  }
}

export function initDiscriminationComparison() {
  const canvas = document.querySelector("#roc-canvas");
  const context = canvas.getContext("2d");
  const stage = document.querySelector("#roc-stage");
  const status = document.querySelector("#roc-status");
  const questionCounter = document.querySelector("#roc-counter");
  const questionCount = document.querySelector("#roc-count");
  const track = document.querySelector("#roc-scroll-track");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  let targetProgress = reducedMotion.matches ? 1 : 0;
  let visibleProgress = targetProgress;
  let animationFrame = 0;
  let currentQuestion = -1;
  let lastMilestone = -1;

  stage.dataset.startGate = "full-viewport";
  stage.dataset.rocProgress = visibleProgress.toFixed(3);
  stage.dataset.visualization = "seven-domain-polar-density-pairs";
  stage.dataset.viewCount = "1";
  stage.dataset.views = "domain-semicircle-density-pairs";
  stage.dataset.coordinateSystem = "seven-polar-semicircles";
  stage.dataset.polarPlotCount = String(domainDiscriminationTrajectories.length);
  stage.dataset.polarDomains = domainDiscriminationTrajectories.map(
    ({ key }) => key,
  ).join(",");
  stage.dataset.polarLayout = "four-by-two-last-row-centered";
  stage.dataset.maximumQuestionLabel = "none";
  stage.dataset.angleLabelFormat = "auroc-only";
  stage.dataset.polarLabelFontDesktop = "13.5";
  stage.dataset.domainTitleFontDesktop = "20";
  stage.dataset.domainTitleFontMobile = "12";
  stage.dataset.domainTitleGapDesktop = "30";
  stage.dataset.domainTitleGapMobile = "24";
  stage.dataset.aurocEndpointLabelGapDesktop = "5";
  stage.dataset.aurocEndpointLabelGapMobile = "12";
  stage.dataset.densityPlotCount = String(domainDiscriminationTrajectories.length);
  stage.dataset.densityCurvesPerDomain = String(discriminationCohorts.length);
  stage.dataset.densityXAxis = "auroc";
  stage.dataset.densityYAxis = "proportion";
  stage.dataset.densityXTicks = "0.5,0.625,0.75,0.875,1.0";
  stage.dataset.densityYTicks = "0,0.5,1";
  stage.dataset.densityXAxisLabelDomains = "spike,lrda";
  stage.dataset.densityYAxisLabelDomains = "spike,lrda";
  stage.dataset.densityProportionLabelFontDesktop = "12";
  stage.dataset.densityProportionLabelFontMobile = "8";
  stage.dataset.densityHeightRatio = "0.5";
  stage.dataset.densityFrame = "left-bottom";
  stage.dataset.densitySync = "visible-question-progress";
  stage.dataset.radialVariable = "questions-asked";
  stage.dataset.angularVariable = "auroc";
  stage.dataset.radianDomain = "0,3.141593";
  stage.dataset.aurocDomain = "0.5,1.0";
  stage.dataset.questionMaximum = String(cortexQuestionMaximum);
  stage.dataset.questionArcValues = [
    0,
    ...innerQuestionArcs,
    cortexQuestionMaximum,
  ].join(",");
  stage.dataset.innerArcStyle = "thin-light-grey";
  stage.dataset.outerArcStyle = "strong";
  stage.dataset.baselineStyle = "strong";
  stage.dataset.cohortCount = String(discriminationCohorts.length);
  stage.dataset.trajectoryCount = String(discriminationTrajectories.length);
  stage.dataset.readersPerDomain = String(readersPerDomain);
  stage.dataset.cohorts = discriminationCohorts.map(({ key }) => key).join(",");
  stage.dataset.cohortColors = discriminationCohorts.map(({ key, color }) => (
    `${key}:${color}`
  )).join(";");
  stage.dataset.cohortSizes = discriminationCohorts.map(({ key, count: size }) => (
    `${key}:${size}`
  )).join(";");
  stage.dataset.experiencedMeanLengthRatio = experiencedMeanStopRatio.toFixed(3);
  stage.dataset.trajectoryMode = "deterministic-question-level";
  stage.dataset.longestTestSource = "first-demonstration-question-horizon";
  stage.dataset.completionGate = "render-before-release";
  stage.dataset.completionHoldDvh = String(completionHoldDvh);
  stage.dataset.completionState = reducedMotion.matches ? "complete" : "running";

  function createPolarPlots(region, isNarrow) {
    const columns = isNarrow ? 2 : 4;
    const rows = isNarrow ? 4 : 2;
    const gapX = isNarrow ? 4 : 8;
    const rowGap = isNarrow ? 8 : 35;
    const topReserve = isNarrow ? 36 : 58;
    const densityTopOffset = isNarrow ? 17 : 28;
    const densityBottomReserve = isNarrow ? 18 : 22;
    const cellWidth = (
      region.right - region.left - gapX * (columns - 1)
    ) / columns;
    const regionHeight = region.bottom - region.top;
    const availablePairHeight = (
      regionHeight - rowGap * (rows - 1)
    ) / rows;
    const radius = Math.min(
      cellWidth * (isNarrow ? .43 : .48),
      (
        availablePairHeight
        - topReserve
        - densityTopOffset
        - densityBottomReserve
      ) / 1.5,
    );
    const densityHeight = radius * .5;
    const pairHeight = (
      topReserve
      + radius
      + densityTopOffset
      + densityHeight
      + densityBottomReserve
    );
    const rowStep = pairHeight + rowGap;
    const groupHeight = pairHeight * rows + rowGap * (rows - 1);
    const firstCenterY = (
      region.top
      + (regionHeight - groupHeight) / 2
      + topReserve
      + radius
    );

    return domainDiscriminationTrajectories.map((domain, index) => {
      const row = Math.floor(index / columns);
      const itemsInRow = Math.min(
        columns,
        domainDiscriminationTrajectories.length - row * columns,
      );
      const rowOffset = (columns - itemsInRow) / 2;
      const column = index % columns;
      const cellLeft = (
        region.left
        + (column + rowOffset) * (cellWidth + gapX)
      );
      const center = {
        x: cellLeft + cellWidth / 2,
        y: firstCenterY + row * rowStep,
      };
      const density = {
        left: center.x - radius,
        top: center.y + densityTopOffset,
        right: center.x + radius,
        bottom: center.y + densityTopOffset + densityHeight,
      };

      return {
        ...domain,
        center,
        radius,
        density,
        bounds: {
          left: center.x - radius,
          top: center.y - radius,
          right: center.x + radius,
          bottom: center.y,
        },
        pairBounds: {
          left: center.x - radius,
          top: center.y - radius - topReserve,
          right: center.x + radius,
          bottom: density.bottom + densityBottomReserve,
        },
      };
    });
  }

  function layout() {
    const { width, height } = canvas.getBoundingClientRect();
    const isNarrow = width < 700;
    const polarRegion = {
      left: isNarrow ? 8 : Math.max(20, width * .016),
      top: height * (isNarrow ? .085 : .09),
      right: width - (isNarrow ? 8 : Math.max(20, width * .016)),
      bottom: height * (isNarrow ? .985 : .98),
    };

    return {
      width,
      height,
      isNarrow,
      panelLayout: isNarrow ? "two-by-four" : "four-by-two",
      polarRegion,
      polarPlots: createPolarPlots(polarRegion, isNarrow),
    };
  }

  function drawQuestionArcs(polar, isNarrow) {
    const grid = cssVar("--figure-grid");
    const ink = cssVar("--figure-ink");
    const serif = cssVar("--research-serif");

    context.save();
    context.strokeStyle = grid;
    context.lineWidth = isNarrow ? .55 : .75;
    context.globalAlpha = .62;
    context.setLineDash([]);

    for (const question of innerQuestionArcs) {
      const radius = polar.radius * question / cortexQuestionMaximum;
      drawSemicirclePath(context, polar.center, radius);
      context.stroke();
    }

    context.globalAlpha = .64;
    context.fillStyle = ink;
    context.font = `${isNarrow ? 7.5 : 11.5}px ${serif}`;
    context.textAlign = "center";
    context.textBaseline = "bottom";
    const labeledArcs = isNarrow ? [200] : innerQuestionArcs;
    for (const question of labeledArcs) {
      const radius = polar.radius * question / cortexQuestionMaximum;
      const point = polarPoint(
        polar.center,
        radius,
        Math.PI * (isNarrow ? .11 : .075),
      );
      context.fillText(String(question), point.x, point.y - 3);
    }
    context.restore();
  }

  function drawOuterFrame(polar, isNarrow) {
    const ink = cssVar("--figure-ink");
    const serif = cssVar("--research-serif");

    context.save();
    context.strokeStyle = ink;
    context.globalAlpha = .78;
    context.lineWidth = isNarrow ? 1.05 : 1.45;
    context.setLineDash([]);
    drawSemicirclePath(context, polar.center, polar.radius);
    context.stroke();

    context.beginPath();
    context.moveTo(polar.bounds.left, polar.center.y);
    context.lineTo(polar.bounds.right, polar.center.y);
    context.stroke();

    context.globalAlpha = 1;
    context.fillStyle = ink;
    context.font = `700 ${isNarrow ? 8.5 : 12.5}px ${serif}`;
    context.textAlign = "center";
    context.textBaseline = "top";
    context.fillText(
      "0 questions",
      polar.center.x,
      polar.center.y + (isNarrow ? 3 : 4),
    );
    context.restore();
  }

  function drawAngleTicks(polar, isNarrow) {
    const ink = cssVar("--figure-ink");
    const serif = cssVar("--research-serif");
    const tickLength = isNarrow ? 3 : 5;
    const labelOffset = isNarrow ? 7 : 11;
    const endpointLabelGap = isNarrow ? 12 : 5;
    const visibleTicks = isNarrow
      ? angleTicks.filter((_, index) => index % 2 === 0)
      : angleTicks;

    context.save();
    context.strokeStyle = ink;
    context.fillStyle = ink;
    context.globalAlpha = .82;
    context.lineWidth = isNarrow ? .7 : .9;
    context.font = `${isNarrow ? 8 : 13.5}px ${serif}`;

    for (const tick of visibleTicks) {
      const inner = polarPoint(polar.center, polar.radius, tick.angle);
      const outer = polarPoint(
        polar.center,
        polar.radius + tickLength,
        tick.angle,
      );
      context.beginPath();
      context.moveTo(inner.x, inner.y);
      context.lineTo(outer.x, outer.y);
      context.stroke();

      if (tick.angle === 0) {
        context.textAlign = "right";
        context.textBaseline = "top";
        context.fillText(
          `${tick.auroc} AUROC`,
          inner.x,
          inner.y + endpointLabelGap,
        );
      } else if (tick.angle === Math.PI) {
        context.textAlign = "left";
        context.textBaseline = "top";
        context.fillText(
          `${tick.auroc} AUROC`,
          inner.x,
          inner.y + endpointLabelGap,
        );
      } else {
        const label = polarPoint(
          polar.center,
          polar.radius + labelOffset,
          tick.angle,
        );
        context.textAlign = "center";
        context.textBaseline = tick.angle === Math.PI / 2 ? "bottom" : "middle";
        context.fillText(tick.auroc, label.x, label.y);
      }
    }
    context.restore();
  }

  function drawDomainLabel(polar, isNarrow) {
    const serif = cssVar("--research-serif");
    context.save();
    context.fillStyle = cssVar(polar.color);
    context.font = `700 ${isNarrow ? 12 : 20}px ${serif}`;
    context.textAlign = "center";
    context.textBaseline = "bottom";
    context.fillText(
      polar.label,
      polar.center.x,
      polar.center.y - polar.radius - (isNarrow ? 24 : 30),
    );
    context.restore();
  }

  function trajectoryAurocAtQuestion(trajectory, visibleQuestion) {
    const bounded = clamp(visibleQuestion, 0, trajectory.stop);
    const lowerIndex = Math.floor(bounded);
    if (lowerIndex >= trajectory.stop) return trajectory.target;

    return lerp(
      trajectory.points[lowerIndex].auroc,
      trajectory.points[lowerIndex + 1].auroc,
      bounded - lowerIndex,
    );
  }

  function createDensityDistributions(polar, visibleQuestion, isNarrow) {
    const sampleCount = isNarrow ? 49 : 81;

    return discriminationCohorts.map((cohort) => {
      const values = polar.trajectories
        .filter(({ cohort: key }) => key === cohort.key)
        .map((trajectory) => (
          trajectoryAurocAtQuestion(trajectory, visibleQuestion)
        ));
      const mean = values.reduce((total, value) => total + value, 0) / values.length;
      const variance = values.reduce(
        (total, value) => total + (value - mean) ** 2,
        0,
      ) / Math.max(1, values.length - 1);
      const deviation = Math.max(.012, Math.sqrt(variance));
      const points = Array.from({ length: sampleCount }, (_, index) => {
        const auroc = .5 + .5 * index / (sampleCount - 1);
        const zScore = (auroc - mean) / deviation;
        return {
          auroc,
          density: Math.exp(-.5 * zScore ** 2) / deviation,
        };
      });

      return {
        ...cohort,
        points,
      };
    });
  }

  function drawDensityAxes(polar, isNarrow) {
    const { density } = polar;
    const ink = cssVar("--figure-ink");
    const serif = cssVar("--research-serif");
    const xTicks = [
      { value: .5, label: "0.5" },
      { value: .625, label: "0.625" },
      { value: .75, label: "0.75" },
      { value: .875, label: "0.875" },
      { value: 1, label: "1.0" },
    ];
    const yTicks = [
      { value: 0, label: "0" },
      { value: .5, label: "0.5" },
      { value: 1, label: "1" },
    ];
    const tickLength = isNarrow ? 2.5 : 4;
    const showAxisTitles = densityAxisTitleDomains.has(polar.key);

    context.save();
    context.strokeStyle = ink;
    context.fillStyle = ink;
    context.globalAlpha = .76;
    context.lineWidth = isNarrow ? .65 : .9;
    context.setLineDash([]);
    context.beginPath();
    context.moveTo(density.left, density.top);
    context.lineTo(density.left, density.bottom);
    context.lineTo(density.right, density.bottom);
    context.stroke();

    context.font = `${isNarrow ? 6.5 : 9.5}px ${serif}`;
    for (const tick of xTicks) {
      const x = lerp(density.left, density.right, (tick.value - .5) / .5);
      context.beginPath();
      context.moveTo(x, density.bottom);
      context.lineTo(x, density.bottom + tickLength);
      context.stroke();
      context.textAlign = tick.value === .5
        ? "left"
        : tick.value === 1
          ? "right"
          : "center";
      context.textBaseline = "top";
      context.fillText(tick.label, x, density.bottom + tickLength + 1);
    }

    for (const tick of yTicks) {
      const y = lerp(density.bottom, density.top, tick.value);
      context.beginPath();
      context.moveTo(density.left - tickLength, y);
      context.lineTo(density.left, y);
      context.stroke();
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(tick.label, density.left - tickLength - 1, y);
    }

    if (showAxisTitles) {
      context.font = `700 ${isNarrow ? 6.5 : 9.5}px ${serif}`;
      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(
        "AUROC",
        (density.left + density.right) / 2,
        density.bottom + (isNarrow ? 11 : 16),
      );

      context.translate(
        density.left - (isNarrow ? 13 : 20),
        (density.top + density.bottom) / 2,
      );
      context.rotate(-Math.PI / 2);
      context.font = `700 ${isNarrow ? 8 : 12}px ${serif}`;
      context.textAlign = "center";
      context.textBaseline = "bottom";
      context.fillText("Proportion", 0, 0);
    }
    context.restore();
  }

  function drawDensityPlot(polar, visibleQuestion, isNarrow) {
    const { density } = polar;
    const distributions = createDensityDistributions(
      polar,
      visibleQuestion,
      isNarrow,
    );
    const maximumDensity = Math.max(
      ...distributions.flatMap(({ points }) => (
        points.map(({ density: value }) => value)
      )),
    );

    for (const distribution of distributions) {
      const mapped = distribution.points.map(({ auroc, density: value }) => ({
        x: lerp(density.left, density.right, (auroc - .5) / .5),
        y: density.bottom - value / maximumDensity * (density.bottom - density.top) * .9,
      }));
      const color = cssVar(distribution.color);

      context.save();
      context.fillStyle = color;
      context.globalAlpha = .055;
      context.beginPath();
      context.moveTo(mapped[0].x, density.bottom);
      mapped.forEach(({ x, y }) => context.lineTo(x, y));
      context.lineTo(mapped.at(-1).x, density.bottom);
      context.closePath();
      context.fill();

      context.strokeStyle = color;
      context.globalAlpha = .78;
      context.lineWidth = isNarrow ? .75 : 1.1;
      context.lineJoin = "round";
      context.lineCap = "round";
      context.setLineDash(
        isNarrow
          ? distribution.dash.map((value) => value * .62)
          : distribution.dash,
      );
      context.beginPath();
      mapped.forEach(({ x, y }, index) => {
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
      context.restore();
    }

    drawDensityAxes(polar, isNarrow);
  }

  function trajectoryPoint(point, polar) {
    const radius = polar.radius * point.question / cortexQuestionMaximum;
    return polarPoint(polar.center, radius, aurocToAngle(point.auroc));
  }

  function drawTrajectory(polar, trajectory, visibleQuestion, isNarrow) {
    const points = revealedTrajectoryPoints(trajectory.points, visibleQuestion);
    if (points.length < 2) return;
    const maximumSegments = isNarrow ? 62 : 105;
    const pointStep = Math.max(1, Math.ceil(points.length / maximumSegments));

    context.save();
    context.strokeStyle = cssVar(trajectory.color);
    context.globalAlpha = trajectory.cohort === "experienced" ? .34 : .4;
    context.lineWidth = isNarrow ? .58 : .84;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.setLineDash(
      isNarrow
        ? trajectory.dash.map((value) => value * .62)
        : trajectory.dash,
    );
    context.beginPath();
    points.forEach((point, index) => {
      if (
        index !== 0
        && index !== points.length - 1
        && index % pointStep !== 0
      ) return;
      const mapped = trajectoryPoint(point, polar);
      if (index === 0) context.moveTo(mapped.x, mapped.y);
      else context.lineTo(mapped.x, mapped.y);
    });
    context.stroke();

    if (visibleQuestion >= trajectory.stop) {
      const endpoint = trajectoryPoint(trajectory.points.at(-1), polar);
      context.globalAlpha = .82;
      context.fillStyle = cssVar(trajectory.color);
      context.beginPath();
      context.arc(
        endpoint.x,
        endpoint.y,
        isNarrow ? .95 : 1.4,
        0,
        Math.PI * 2,
      );
      context.fill();
    }
    context.restore();
  }

  function drawOrigin(polar, isNarrow) {
    context.save();
    context.fillStyle = cssVar("--figure-ink");
    context.globalAlpha = .76;
    context.beginPath();
    context.arc(
      polar.center.x,
      polar.center.y,
      isNarrow ? 1.8 : 2.3,
      0,
      Math.PI * 2,
    );
    context.fill();
    context.restore();
  }

  function drawPolarSmallMultiple(graph, polar) {
    const visibleQuestion = visibleProgress * cortexQuestionMaximum;
    drawQuestionArcs(polar, graph.isNarrow);
    for (const cohort of discriminationCohorts) {
      for (const trajectory of polar.trajectories) {
        if (trajectory.cohort === cohort.key) {
          drawTrajectory(
            polar,
            trajectory,
            visibleQuestion,
            graph.isNarrow,
          );
        }
      }
    }
    drawOuterFrame(polar, graph.isNarrow);
    drawAngleTicks(polar, graph.isNarrow);
    drawOrigin(polar, graph.isNarrow);
    drawDomainLabel(polar, graph.isNarrow);
    drawDensityPlot(polar, visibleQuestion, graph.isNarrow);
  }

  function drawPolarDiscrimination(graph) {
    for (const polar of graph.polarPlots) {
      drawPolarSmallMultiple(graph, polar);
    }
  }

  function draw() {
    const graph = layout();
    context.clearRect(0, 0, graph.width, graph.height);

    const polarBounds = [
      Math.min(...graph.polarPlots.map(({ bounds }) => bounds.left)),
      Math.min(...graph.polarPlots.map(({ bounds }) => bounds.top)),
      Math.max(...graph.polarPlots.map(({ bounds }) => bounds.right)),
      Math.max(...graph.polarPlots.map(({ bounds }) => bounds.bottom)),
    ];
    const individualPolarBounds = graph.polarPlots.map(({ bounds }) => (
      [bounds.left, bounds.top, bounds.right, bounds.bottom]
        .map((value) => value.toFixed(1))
        .join(",")
    ));
    const densityBounds = [
      Math.min(...graph.polarPlots.map(({ density }) => density.left)),
      Math.min(...graph.polarPlots.map(({ density }) => density.top)),
      Math.max(...graph.polarPlots.map(({ density }) => density.right)),
      Math.max(...graph.polarPlots.map(({ density }) => density.bottom)),
    ];
    const individualDensityBounds = graph.polarPlots.map(({ density }) => (
      [density.left, density.top, density.right, density.bottom]
        .map((value) => value.toFixed(1))
        .join(",")
    ));
    const pairBounds = [
      Math.min(...graph.polarPlots.map(({ pairBounds: bounds }) => bounds.left)),
      Math.min(...graph.polarPlots.map(({ pairBounds: bounds }) => bounds.top)),
      Math.max(...graph.polarPlots.map(({ pairBounds: bounds }) => bounds.right)),
      Math.max(...graph.polarPlots.map(({ pairBounds: bounds }) => bounds.bottom)),
    ];
    const individualPairBounds = graph.polarPlots.map(
      ({ pairBounds: bounds }) => (
        [bounds.left, bounds.top, bounds.right, bounds.bottom]
          .map((value) => value.toFixed(1))
          .join(",")
      ),
    );
    stage.dataset.panelLayout = graph.panelLayout;
    stage.dataset.polarPlotBounds = (
      polarBounds.map((value) => value.toFixed(1)).join(",")
    );
    stage.dataset.polarPlotsBounds = individualPolarBounds.join(";");
    stage.dataset.densityPlotBounds = (
      densityBounds.map((value) => value.toFixed(1)).join(",")
    );
    stage.dataset.densityPlotsBounds = individualDensityBounds.join(";");
    stage.dataset.pairedPlotBounds = (
      pairBounds.map((value) => value.toFixed(1)).join(",")
    );
    stage.dataset.pairedPlotsBounds = individualPairBounds.join(";");
    stage.dataset.plotContained = String(
      pairBounds[0] >= 0
      && pairBounds[1] >= 0
      && pairBounds[2] <= graph.width
      && pairBounds[3] <= graph.height
    );

    drawPolarDiscrimination(graph);
  }

  function syncCounters() {
    const nextQuestion = Math.round(visibleProgress * cortexQuestionMaximum);

    if (nextQuestion !== currentQuestion) {
      currentQuestion = nextQuestion;
      questionCount.textContent = String(currentQuestion);
      questionCounter.setAttribute(
        "aria-label",
        `${currentQuestion} of ${cortexQuestionMaximum} questions asked`,
      );
    }
  }

  function syncCompletionState() {
    const isComplete = (
      currentQuestion === cortexQuestionMaximum
      && visibleProgress >= .9995
    );
    stage.dataset.completionState = isComplete ? "complete" : "running";
  }

  function announceProgress() {
    const milestone = Math.round(visibleProgress * 4);
    if (milestone === lastMilestone) return;
    lastMilestone = milestone;

    if (milestone === 0) {
      status.textContent = (
        "The seven CORTEX discrimination and density graph pairs are ready."
      );
    } else if (milestone === 4) {
      status.textContent = (
        "All simulated reader trajectories and AUROC densities are fully revealed."
      );
    } else {
      status.textContent = `${currentQuestion} questions revealed.`;
    }
  }

  function renderVisibleState() {
    stage.dataset.rocProgress = visibleProgress.toFixed(3);
    stage.dataset.visibleQuestion = (
      visibleProgress * cortexQuestionMaximum
    ).toFixed(1);
    syncCounters();
    syncCompletionState();
    announceProgress();
    draw();
  }

  function animate() {
    animationFrame = 0;
    const difference = targetProgress - visibleProgress;

    if (Math.abs(difference) < .0004) {
      visibleProgress = targetProgress;
      renderVisibleState();
      return;
    }

    visibleProgress += difference * .095;
    renderVisibleState();
    animationFrame = requestAnimationFrame(animate);
  }

  function setScrollProgress(progress) {
    const bounded = clamp(progress);
    targetProgress = reducedMotion.matches ? 1 : bounded;
    stage.dataset.scrollProgress = bounded.toFixed(3);

    if (bounded >= .999 || reducedMotion.matches) {
      targetProgress = 1;
      visibleProgress = 1;
      if (animationFrame) cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      renderVisibleState();
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
  syncCounters();
  setScrollProgress(reducedMotion.matches ? 1 : 0);

  return {
    redraw: canvasObserver.redraw,
  };
}
