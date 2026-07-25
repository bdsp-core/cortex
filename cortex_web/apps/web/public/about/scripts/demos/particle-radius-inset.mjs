import {
  createParticleRadiusInterpolator,
  mulberry32,
  slowCollapseProgress,
} from "../../demo-core.mjs?v=20260724-radius-inset-optimized";
import {
  cssVar,
  lerp,
  scale,
} from "../shared/chart.mjs";

const radiusTickCount = 3;
const insetScale = .75;
const snapshotBlend = .9;
const snapshotCorrelation = .2;
const snapshotEnvelopePower = .7;
const snapshotParticleLimit = 500;

function parametricSnapshotRadius(domain, baselineRadius, question, sampleCount) {
  const random = mulberry32((
    domain.seed
    ^ Math.imul(question + 1, 0x9e3779b1)
    ^ 0x85ebca6b
  ) >>> 0);
  const first = Math.max(1e-12, random());
  const second = random();
  const gaussian = Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
  const standardError = baselineRadius / Math.sqrt(2 * sampleCount);

  // A parametric bootstrap preserves the sampling scale of a 500-particle
  // cloud without resampling all 500 points for every one of 251 frames.
  return Math.max(0, baselineRadius + gaussian * standardError * .68);
}

function createStochasticRadiusPoints(domain, totalQuestions, radiusAt) {
  const sampleCount = Math.min(snapshotParticleLimit, domain.prior.length);
  let previousDeviation = 0;

  const points = Array.from({ length: totalQuestions + 1 }, (_, question) => {
    const questionProgress = question / totalQuestions;
    const collapseProgress = slowCollapseProgress(questionProgress);
    const baselineRadius = radiusAt(collapseProgress);

    if (question === 0 || question === totalQuestions) {
      return {
        question,
        radius: baselineRadius,
        baselineRadius,
      };
    }

    const snapshotRadius = parametricSnapshotRadius(
      domain,
      baselineRadius,
      question,
      sampleCount,
    );
    const innovation = snapshotRadius - baselineRadius;
    previousDeviation = (
      snapshotCorrelation * previousDeviation
      + (1 - snapshotCorrelation) * innovation
    );
    const envelope = Math.sin(Math.PI * questionProgress) ** snapshotEnvelopePower;

    return {
      question,
      radius: Math.max(
        0,
        baselineRadius + previousDeviation * snapshotBlend * envelope,
      ),
      baselineRadius,
    };
  });

  const upwardSteps = points.reduce((count, point, index) => (
    index > 0 && point.radius > points[index - 1].radius ? count + 1 : count
  ), 0);
  const maximumRelativeDeviation = Math.max(
    ...points.map(({ radius, baselineRadius }) => (
      Math.abs(radius - baselineRadius) / Math.max(1e-12, baselineRadius)
    )),
  );

  return {
    points,
    sampleCount,
    upwardSteps,
    maximumRelativeDeviation,
  };
}

export function createDomainRadiusModel(domainClouds, totalQuestions) {
  const curves = domainClouds.map((domain) => {
    const radiusAt = createParticleRadiusInterpolator(
      domain.prior,
      domain.posterior,
    );
    const trajectory = createStochasticRadiusPoints(
      domain,
      totalQuestions,
      radiusAt,
    );
    return {
      key: domain.key,
      label: domain.label,
      color: domain.color,
      ...trajectory,
    };
  });
  const largestRadius = Math.max(
    ...curves.flatMap(({ points }) => points.map(({ radius }) => radius)),
  );
  const yMaximum = Math.max(.2, Math.ceil(largestRadius / .2) * .2);

  return {
    curves,
    totalQuestions,
    trajectoryMode: "deterministic-parametric-bootstrap",
    snapshotParticles: Math.min(...curves.map(({ sampleCount }) => sampleCount)),
    minimumUpwardSteps: Math.min(...curves.map(({ upwardSteps }) => upwardSteps)),
    maximumUpwardSteps: Math.max(...curves.map(({ upwardSteps }) => upwardSteps)),
    maximumRelativeDeviation: Math.max(
      ...curves.map((curve) => curve.maximumRelativeDeviation),
    ),
    yMaximum,
    yTicks: Array.from(
      { length: radiusTickCount + 1 },
      (_, index) => yMaximum * index / radiusTickCount,
    ),
  };
}

function insetGeometry(graph) {
  const plotRight = graph.width - graph.right;
  const plotBottom = graph.height - graph.bottom;
  const zeroX = scale(0, [-2, 2], [graph.left, plotRight]);
  const zeroY = scale(0, [-2, 2], [plotBottom, graph.top]);
  const isNarrow = graph.width < 600;
  const quadrantWidth = plotRight - zeroX;
  const quadrantHeight = plotBottom - zeroY;
  const fullOuter = {
    left: zeroX + Math.max(4, quadrantWidth * .025),
    top: zeroY + Math.max(8, quadrantHeight * .035),
    right: plotRight - Math.max(7, quadrantWidth * .025),
    bottom: plotBottom - Math.max(7, quadrantHeight * .025),
  };
  const fullOuterWidth = fullOuter.right - fullOuter.left;
  const fullOuterHeight = fullOuter.bottom - fullOuter.top;
  const fullPlot = {
    left: fullOuter.left + Math.min(isNarrow ? 32 : 52, fullOuterWidth * .29),
    top: fullOuter.top + Math.min(isNarrow ? 25 : 35, fullOuterHeight * .22),
    right: fullOuter.right - Math.min(isNarrow ? 10 : 13, fullOuterWidth * .08),
    bottom: fullOuter.bottom - Math.min(isNarrow ? 37 : 48, fullOuterHeight * .29),
  };
  const scaleX = (value) => fullOuter.right - (fullOuter.right - value) * insetScale;
  const scaleY = (value) => fullOuter.bottom - (fullOuter.bottom - value) * insetScale;
  const outer = {
    left: scaleX(fullOuter.left),
    top: scaleY(fullOuter.top),
    right: fullOuter.right,
    bottom: fullOuter.bottom,
  };
  const plot = {
    left: scaleX(fullPlot.left),
    top: scaleY(fullPlot.top),
    right: scaleX(fullPlot.right),
    bottom: scaleY(fullPlot.bottom),
  };

  return {
    isNarrow,
    quadrant: {
      left: zeroX,
      top: zeroY,
      right: plotRight,
      bottom: plotBottom,
    },
    outer,
    plot: {
      ...plot,
      width: Math.max(1, plot.right - plot.left),
      height: Math.max(1, plot.bottom - plot.top),
    },
  };
}

function revealedPoints(points, threshold) {
  const lastQuestion = Math.min(
    points.length - 1,
    Math.max(0, Math.floor(threshold)),
  );
  const visible = points.slice(0, lastQuestion + 1);

  if (threshold <= lastQuestion || lastQuestion >= points.length - 1) {
    return visible;
  }

  const current = points[lastQuestion];
  const next = points[lastQuestion + 1];
  const progress = threshold - lastQuestion;
  visible.push({
    question: threshold,
    radius: lerp(current.radius, next.radius, progress),
  });
  return visible;
}

function pointToCanvas(point, model, plot) {
  return {
    x: scale(
      point.question,
      [0, model.totalQuestions],
      [plot.left, plot.right],
    ),
    y: scale(
      point.radius,
      [0, model.yMaximum],
      [plot.bottom, plot.top],
    ),
  };
}

function drawInsetAxes(context, model, inset) {
  const { isNarrow, outer, plot } = inset;
  const ink = cssVar("--figure-ink");
  const serif = cssVar("--research-serif");
  const xTicks = [0, model.totalQuestions / 2, model.totalQuestions];

  context.save();
  context.strokeStyle = ink;
  context.lineWidth = isNarrow ? 1 : 1.25;
  context.beginPath();
  context.moveTo(plot.left, plot.top);
  context.lineTo(plot.left, plot.bottom);
  context.lineTo(plot.right, plot.bottom);
  context.stroke();

  context.fillStyle = ink;
  context.font = `${isNarrow ? 8 : 11}px ${serif}`;
  context.textAlign = "center";
  context.textBaseline = "top";
  for (const tick of xTicks) {
    const x = scale(tick, [0, model.totalQuestions], [plot.left, plot.right]);
    context.fillText(String(Math.round(tick)), x, plot.bottom + 5);
  }

  context.textAlign = "right";
  context.textBaseline = "middle";
  for (const tick of model.yTicks) {
    const y = scale(tick, [0, model.yMaximum], [plot.bottom, plot.top]);
    context.fillText(tick.toFixed(1), plot.left - (isNarrow ? 4 : 7), y);
  }

  context.font = `700 ${isNarrow ? 8.5 : 12}px ${serif}`;
  context.textAlign = "center";

  context.save();
  context.translate(
    outer.left + (isNarrow ? 6 : 9),
    plot.top + plot.height / 2,
  );
  context.rotate(-Math.PI / 2);
  context.textBaseline = "middle";
  context.fillText("RMS radius", 0, 0);
  context.restore();
  context.restore();
}

function drawRadiusCurve(context, model, inset, curve, threshold) {
  const points = revealedPoints(curve.points, threshold);
  if (!points.length) return;

  const color = cssVar(curve.color);
  context.strokeStyle = color;
  context.lineWidth = inset.isNarrow ? 1.35 : 2.15;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.setLineDash([]);
  context.beginPath();
  points.forEach((point, index) => {
    const mapped = pointToCanvas(point, model, inset.plot);
    if (index === 0) context.moveTo(mapped.x, mapped.y);
    else context.lineTo(mapped.x, mapped.y);
  });
  context.stroke();

  const endpoint = pointToCanvas(points.at(-1), model, inset.plot);
  context.fillStyle = color;
  context.beginPath();
  context.arc(
    endpoint.x,
    endpoint.y,
    inset.isNarrow ? 1.8 : 3,
    0,
    Math.PI * 2,
  );
  context.fill();
}

export function drawParticleRadiusInset({
  context,
  graph,
  model,
  progress,
  stage,
}) {
  const inset = insetGeometry(graph);
  const threshold = progress * model.totalQuestions;
  const { quadrant, outer, plot } = inset;
  const isContained = (
    outer.left > quadrant.left
    && outer.top > quadrant.top
    && outer.right < quadrant.right
    && outer.bottom < quadrant.bottom
  );

  stage.dataset.radiusInsetBounds = [
    outer.left,
    outer.top,
    outer.right,
    outer.bottom,
  ].map((value) => value.toFixed(1)).join(",");
  stage.dataset.radiusInsetQuadrantBounds = [
    quadrant.left,
    quadrant.top,
    quadrant.right,
    quadrant.bottom,
  ].map((value) => value.toFixed(1)).join(",");
  stage.dataset.radiusInsetContained = String(isContained);
  stage.dataset.radiusInsetFrame = "left-bottom";
  stage.dataset.radiusInsetGrid = "none";
  stage.dataset.radiusInsetScale = String(insetScale);
  stage.dataset.radiusInsetTitle = "none";
  stage.dataset.radiusInsetXAxisLabel = "none";
  stage.dataset.radiusRevealQuestion = threshold.toFixed(1);

  context.save();
  drawInsetAxes(context, model, inset);

  context.beginPath();
  context.rect(plot.left, plot.top, plot.width, plot.height);
  context.clip();
  for (const curve of model.curves) {
    drawRadiusCurve(context, model, inset, curve, threshold);
  }
  context.restore();
}
