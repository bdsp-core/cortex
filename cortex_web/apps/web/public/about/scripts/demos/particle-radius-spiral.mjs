import {
  mulberry32,
} from "../../demo-core.mjs?v=20260724-radius-inset-optimized";
import {
  clamp,
  cssVar,
  lerp,
  scale,
} from "../shared/chart.mjs";

const spiralTurns = 4.75;
const ringFractions = [.25, .5, .75, 1];
const radialCorrelation = .84;
const radialVarianceScale = .038;
const radialVarianceLimit = .065;
const angularCorrelation = .7;
const angularVarianceScale = .15;
const pathVarianceSeed = 0x5a17c9e3;
export const spiralCenterBias = -1.5;
export const spiralTitlePlacement = "bottom";

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function gaussian(random) {
  const first = Math.max(1e-12, random());
  const second = random();
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}

function correlatedSeries(length, correlation, random) {
  const innovationScale = Math.sqrt(1 - correlation ** 2);
  let state = 0;

  return Array.from({ length }, () => {
    state = correlation * state + innovationScale * gaussian(random);
    return state;
  });
}

function createVariableAngles(totalQuestions, random) {
  const correlated = correlatedSeries(
    totalQuestions,
    angularCorrelation,
    random,
  );
  const rawSteps = correlated.map((value) => (
    clamp(1 + value * angularVarianceScale, .78, 1.22)
  ));
  const stepTotal = rawSteps.reduce((total, value) => total + value, 0);
  const normalizedSteps = rawSteps.map(
    (value) => value * totalQuestions / stepTotal,
  );
  let cumulativeStep = 0;
  const angles = [-Math.PI / 2];

  normalizedSteps.forEach((step) => {
    cumulativeStep += step;
    angles.push(
      -Math.PI / 2
      + cumulativeStep / totalQuestions * spiralTurns * Math.PI * 2,
    );
  });

  return {
    angles,
    minimumStepRatio: Math.min(...normalizedSteps),
    maximumStepRatio: Math.max(...normalizedSteps),
  };
}

function chooseQuestionDomain(curves, question, allocations, counts) {
  const radii = curves.map(({ points }) => points[question - 1].radius);
  const radiusTotal = radii.reduce((total, radius) => total + radius, 0);

  radii.forEach((radius, index) => {
    allocations[index] += radius / radiusTotal;
  });

  let selectedIndex = 0;
  let largestDeficit = -Infinity;
  allocations.forEach((allocation, index) => {
    const deficit = allocation - counts[index];
    if (deficit > largestDeficit) {
      largestDeficit = deficit;
      selectedIndex = index;
    }
  });
  counts[selectedIndex] += 1;
  return curves[selectedIndex];
}

export function createMeanRadiusSpiralModel(radiusModel) {
  const { curves, totalQuestions } = radiusModel;
  const meanRadii = Array.from({ length: totalQuestions + 1 }, (_, question) => (
    mean(curves.map(({ points }) => points[question].radius))
  ));
  const resolvedMeanRadius = meanRadii.at(-1);
  const maximumRemainingRadius = Math.max(
    1e-12,
    meanRadii[0] - resolvedMeanRadius,
  );
  const allocations = Array(curves.length).fill(0);
  const counts = Array(curves.length).fill(0);
  const random = mulberry32(pathVarianceSeed);
  const radialStates = correlatedSeries(
    totalQuestions + 1,
    radialCorrelation,
    random,
  );
  const variableAngles = createVariableAngles(totalQuestions, random);
  const baselineNormalizedRadii = meanRadii.map((meanRadius) => clamp(
    (meanRadius - resolvedMeanRadius) / maximumRemainingRadius,
  ));

  const points = meanRadii.map((meanRadius, question) => {
    const questionProgress = question / totalQuestions;
    const baselineNormalizedRadius = baselineNormalizedRadii[question];
    const envelope = Math.sin(Math.PI * questionProgress) ** .72;
    const radialVariance = (
      question === 0 || question === totalQuestions
        ? 0
        : clamp(
          radialStates[question]
            * radialVarianceScale
            * envelope
            * (.45 + .55 * baselineNormalizedRadius),
          -radialVarianceLimit,
          radialVarianceLimit,
        )
    );
    const normalizedRadius = clamp(
      baselineNormalizedRadius + radialVariance,
    );
    const base = {
      question,
      angle: variableAngles.angles[question],
      meanRadius,
      remainingRadius: Math.max(0, meanRadius - resolvedMeanRadius),
      baselineNormalizedRadius,
      radialVariance,
      normalizedRadius,
    };

    if (question === 0) {
      return {
        ...base,
        domain: null,
        color: null,
      };
    }

    const domain = chooseQuestionDomain(
      curves,
      question,
      allocations,
      counts,
    );
    return {
      ...base,
      domain: domain.key,
      color: domain.color,
    };
  });
  const maximumNormalizedVariance = Math.max(
    ...points.map(({ radialVariance }) => Math.abs(radialVariance)),
  );
  const radialReversalCount = points.reduce((count, point, index) => (
    index > 0 && point.normalizedRadius > points[index - 1].normalizedRadius
      ? count + 1
      : count
  ), 0);

  return {
    points,
    totalQuestions,
    turns: spiralTurns,
    guideRingCount: ringFractions.length,
    maximumRemainingRadius,
    resolvedMeanRadius,
    questionCounts: Object.fromEntries(
      curves.map(({ key }, index) => [key, counts[index]]),
    ),
    pathMode: "deterministic-correlated-radius-and-angle-variance",
    maximumNormalizedVariance,
    radialReversalCount,
    angularStepRatio: {
      minimum: variableAngles.minimumStepRatio,
      maximum: variableAngles.maximumStepRatio,
    },
    scheduleMode: "uncertainty-weighted-fair-allocation",
    radialMetric: "mean-rms-radius-above-resolved-endpoint",
  };
}

function spiralGeometry(graph) {
  const plotRight = graph.width - graph.right;
  const plotBottom = graph.height - graph.bottom;
  const zeroX = scale(0, [-2, 2], [graph.left, plotRight]);
  const zeroY = scale(0, [-2, 2], [plotBottom, graph.top]);
  const isNarrow = graph.width < 600;
  const quadrant = {
    left: graph.left,
    top: zeroY,
    right: zeroX,
    bottom: plotBottom,
  };
  const quadrantWidth = quadrant.right - quadrant.left;
  const quadrantHeight = quadrant.bottom - quadrant.top;
  const titleReserve = isNarrow ? 15 : 22;
  const containmentMargin = isNarrow ? 2 : 5;
  const center = {
    x: scale(
      spiralCenterBias,
      [-2, 2],
      [graph.left, plotRight],
    ),
    y: quadrant.top + quadrantHeight * .55,
  };
  const radius = Math.min(
    quadrantWidth * .36,
    quadrantHeight * .38,
    center.x - quadrant.left - containmentMargin,
    center.y - quadrant.top - containmentMargin,
    quadrant.bottom - center.y - titleReserve - containmentMargin,
  );

  return {
    isNarrow,
    center,
    radius,
    quadrant,
    outer: {
      left: center.x - radius,
      top: center.y - radius,
      right: center.x + radius,
      bottom: center.y + radius + titleReserve,
    },
  };
}

function pointToCanvas(point, spiral) {
  const radius = point.normalizedRadius * spiral.radius;
  return {
    x: spiral.center.x + Math.cos(point.angle) * radius,
    y: spiral.center.y + Math.sin(point.angle) * radius,
  };
}

function visibleSpiralPoints(points, threshold) {
  const lastQuestion = Math.min(
    points.length - 1,
    Math.max(0, Math.floor(threshold)),
  );
  const visible = points.slice(0, lastQuestion + 1);

  if (lastQuestion >= points.length - 1 || threshold === lastQuestion) {
    return visible;
  }

  const current = points[lastQuestion];
  const next = points[lastQuestion + 1];
  const progress = threshold - lastQuestion;
  visible.push({
    question: threshold,
    angle: lerp(current.angle, next.angle, progress),
    normalizedRadius: lerp(
      current.normalizedRadius,
      next.normalizedRadius,
      progress,
    ),
  });
  return visible;
}

function drawSpiralGuide(context, model, spiral) {
  const grid = cssVar("--figure-grid");
  const ink = cssVar("--figure-ink");
  const serif = cssVar("--research-serif");

  context.save();
  context.strokeStyle = grid;
  context.globalAlpha = .5;
  context.lineWidth = spiral.isNarrow ? .55 : .8;
  context.setLineDash([]);
  for (const fraction of ringFractions) {
    context.beginPath();
    context.arc(
      spiral.center.x,
      spiral.center.y,
      spiral.radius * fraction,
      0,
      Math.PI * 2,
    );
    context.stroke();
  }

  context.globalAlpha = .72;
  context.fillStyle = ink;
  context.font = `${spiral.isNarrow ? 7 : 10.5}px ${serif}`;
  context.textAlign = "left";
  context.textBaseline = "top";
  for (const fraction of ringFractions) {
    context.fillText(
      (model.maximumRemainingRadius * fraction).toFixed(2),
      spiral.center.x + 3,
      spiral.center.y - spiral.radius * fraction + 2,
    );
  }
  context.fillText("0", spiral.center.x + 4, spiral.center.y + 3);

  context.globalAlpha = .86;
  context.font = `700 ${spiral.isNarrow ? 8.5 : 12}px ${serif}`;
  context.textAlign = "center";
  context.textBaseline = "top";
  context.fillText(
    "Mean RMS remaining",
    spiral.center.x,
    spiral.center.y + spiral.radius + (spiral.isNarrow ? 4 : 7),
  );
  context.restore();
}

function drawSpiralQuestions(context, model, spiral, threshold) {
  const visible = visibleSpiralPoints(model.points, threshold);
  const mapped = visible.map((point) => pointToCanvas(point, spiral));
  const ink = cssVar("--figure-ink");

  if (mapped.length > 1) {
    context.save();
    context.strokeStyle = ink;
    context.globalAlpha = .26;
    context.lineWidth = spiral.isNarrow ? .7 : 1.05;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.setLineDash([]);
    context.beginPath();
    mapped.forEach((point, index) => {
      if (index === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    });
    context.stroke();
    context.restore();
  }

  const lastCompleteQuestion = Math.floor(threshold);
  const pointRadius = spiral.isNarrow ? 1.45 : 2.25;
  context.save();
  context.globalAlpha = .9;
  for (let question = 1; question <= lastCompleteQuestion; question += 1) {
    const point = pointToCanvas(model.points[question], spiral);
    context.fillStyle = cssVar(model.points[question].color);
    context.beginPath();
    context.arc(point.x, point.y, pointRadius, 0, Math.PI * 2);
    context.fill();
  }
  context.restore();
}

export function drawMeanRadiusSpiral({
  context,
  graph,
  model,
  progress,
  stage,
}) {
  const spiral = spiralGeometry(graph);
  const threshold = progress * model.totalQuestions;
  const { outer, quadrant } = spiral;
  const isContained = (
    outer.left > quadrant.left
    && outer.top > quadrant.top
    && outer.right < quadrant.right
    && outer.bottom < quadrant.bottom
  );

  stage.dataset.spiralInsetBounds = [
    outer.left,
    outer.top,
    outer.right,
    outer.bottom,
  ].map((value) => value.toFixed(1)).join(",");
  stage.dataset.spiralInsetQuadrantBounds = [
    quadrant.left,
    quadrant.top,
    quadrant.right,
    quadrant.bottom,
  ].map((value) => value.toFixed(1)).join(",");
  stage.dataset.spiralInsetContained = String(isContained);
  stage.dataset.spiralRevealQuestion = threshold.toFixed(1);

  drawSpiralGuide(context, model, spiral);
  drawSpiralQuestions(context, model, spiral, threshold);
}
