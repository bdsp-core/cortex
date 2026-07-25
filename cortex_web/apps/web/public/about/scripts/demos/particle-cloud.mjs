import {
  createParticles,
  recenterParticles,
  slowCollapseProgress,
  updateTestParticles,
  weightedStats,
} from "../../demo-core.mjs?v=20260724-radius-inset-optimized";
import {
  createMarkerController,
  cssVar,
  lerp,
  observeCanvas,
  scale,
} from "../shared/chart.mjs";
import { eegDomains } from "../data/domains.mjs";
import {
  createDomainRadiusModel,
  drawParticleRadiusInset,
} from "./particle-radius-inset.mjs?v=20260724-parametric-radius";
import {
  createMeanRadiusSpiralModel,
  drawMeanRadiusSpiral,
  spiralCenterBias,
  spiralTitlePlacement,
} from "./particle-radius-spiral.mjs?v=20260724-spiral-position";

const biasDomain = [-2, 2];
const skillDomain = [-2, 2];
const axisTicks = [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2];
const particleCount = 500;
const particleRadius = { narrow: 1.15, wide: 1.6 };
const particleOpacity = { prior: .48, posterior: .68 };
const posteriorUpdates = 7;
const totalQuestions = 250;
const completionHoldDvh = 30;
const questionProgressEase = .06;
const particleCollapseEase = .075;

function createDomainCloud(domain) {
  const prior = recenterParticles(createParticles(particleCount, domain.seed));
  let posterior = prior;

  for (let question = 1; question <= posteriorUpdates; question += 1) {
    posterior = updateTestParticles(posterior, domain.target, question).particles;
  }

  return {
    ...domain,
    prior,
    posterior,
    priorStats: weightedStats(prior),
    posteriorStats: weightedStats(posterior),
  };
}

function yieldToBrowser() {
  return new Promise((resolve) => window.setTimeout(resolve, 0));
}

export async function initParticleCloud() {
  const canvas = document.querySelector("#particle-canvas");
  const context = canvas.getContext("2d");
  const stage = document.querySelector("#graph-stage");
  const status = document.querySelector("#graph-status");
  const questionCounter = document.querySelector("#question-counter");
  const questionCount = document.querySelector("#question-count");
  const track = document.querySelector("#scroll-track");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const domainClouds = [];

  for (const domain of eegDomains) {
    domainClouds.push(createDomainCloud(domain));
    if (!reducedMotion.matches) await yieldToBrowser();
  }

  const radiusModel = createDomainRadiusModel(domainClouds, totalQuestions);
  const spiralModel = createMeanRadiusSpiralModel(radiusModel);

  stage.dataset.particlesPerDomain = String(particleCount);
  stage.dataset.particleShape = "circle";
  stage.dataset.particleRadiusNarrow = String(particleRadius.narrow);
  stage.dataset.particleRadiusWide = String(particleRadius.wide);
  stage.dataset.particleOpacityPrior = String(particleOpacity.prior);
  stage.dataset.particleOpacityPosterior = String(particleOpacity.posterior);
  stage.dataset.axisStyle = "zero-axes-only";
  stage.dataset.axisDomain = "-2,2";
  stage.dataset.axisIncrement = "0.5";
  stage.dataset.yAxisLabelPlacement = "tick-relative";
  stage.dataset.guideLines = "none";
  stage.dataset.questionTotal = String(totalQuestions);
  stage.dataset.radiusInset = "domain-particle-radius";
  stage.dataset.radiusInsetBackground = "transparent";
  stage.dataset.radiusMetric = "root-mean-square-distance";
  stage.dataset.radiusCurveCount = String(radiusModel.curves.length);
  stage.dataset.radiusQuestionTotal = String(radiusModel.totalQuestions);
  stage.dataset.radiusYMaximum = radiusModel.yMaximum.toFixed(1);
  stage.dataset.radiusSync = "visible-question-progress";
  stage.dataset.radiusTrajectoryMode = radiusModel.trajectoryMode;
  stage.dataset.radiusSnapshotParticles = String(radiusModel.snapshotParticles);
  stage.dataset.radiusUpwardSteps = [
    radiusModel.minimumUpwardSteps,
    radiusModel.maximumUpwardSteps,
  ].join(",");
  stage.dataset.radiusMaximumRelativeDeviation = (
    radiusModel.maximumRelativeDeviation.toFixed(4)
  );
  stage.dataset.spiralInset = "mean-rms-question-decay";
  stage.dataset.spiralInsetQuadrant = "three";
  stage.dataset.spiralInsetBackground = "transparent";
  stage.dataset.spiralCenterBias = String(spiralCenterBias);
  stage.dataset.spiralTitlePlacement = spiralTitlePlacement;
  stage.dataset.spiralMetric = spiralModel.radialMetric;
  stage.dataset.spiralPointCount = String(spiralModel.totalQuestions);
  stage.dataset.spiralQuestionTotal = String(spiralModel.totalQuestions);
  stage.dataset.spiralTurns = String(spiralModel.turns);
  stage.dataset.spiralGuideRingCount = String(spiralModel.guideRingCount);
  stage.dataset.spiralDomainEncoding = "selected-domain-color";
  stage.dataset.spiralScheduleMode = spiralModel.scheduleMode;
  stage.dataset.spiralPathMode = spiralModel.pathMode;
  stage.dataset.spiralMaximumNormalizedVariance = (
    spiralModel.maximumNormalizedVariance.toFixed(4)
  );
  stage.dataset.spiralRadialReversalCount = String(
    spiralModel.radialReversalCount,
  );
  stage.dataset.spiralAngularStepRatio = [
    spiralModel.angularStepRatio.minimum.toFixed(3),
    spiralModel.angularStepRatio.maximum.toFixed(3),
  ].join(",");
  stage.dataset.spiralQuestionCounts = Object.entries(
    spiralModel.questionCounts,
  ).map(([key, count]) => `${key}:${count}`).join(",");
  stage.dataset.spiralEndpoint = "origin";
  stage.dataset.spiralSync = "visible-question-progress";
  stage.dataset.spiralResolvedMeanRadius = (
    spiralModel.resolvedMeanRadius.toFixed(4)
  );
  stage.dataset.spiralMaximumRemainingRadius = (
    spiralModel.maximumRemainingRadius.toFixed(4)
  );
  stage.dataset.completionGate = "render-before-release";
  stage.dataset.completionTransition = "eased";
  stage.dataset.completionHoldDvh = String(completionHoldDvh);
  stage.dataset.completionState = reducedMotion.matches ? "complete" : "running";
  stage.dataset.initialMean = "0,0";
  stage.dataset.initialMeanMaximumError = String(Math.max(
    ...domainClouds.map(({ priorStats }) => Math.hypot(priorStats.skill, priorStats.bias)),
  ));

  let targetProgress = reducedMotion.matches ? 1 : 0;
  let visibleProgress = targetProgress;
  let targetCollapse = reducedMotion.matches ? 1 : 0;
  let visibleCollapse = targetCollapse;
  let animationFrame = 0;
  let lastMilestone = -1;
  let currentQuestionCount = -1;

  function geometry() {
    const rect = canvas.getBoundingClientRect();
    const plotTop = rect.width <= 700
      ? Math.max(104, rect.height * .14)
      : Math.max(96, rect.height * .12);

    return {
      width: rect.width,
      height: rect.height,
      left: Math.max(52, rect.width * .075),
      right: Math.max(20, rect.width * .035),
      top: plotTop,
      bottom: Math.max(48, rect.height * .095),
    };
  }

  function graphPoint(point, graph) {
    return {
      x: scale(point.bias, biasDomain, [graph.left, graph.width - graph.right]),
      y: scale(point.skill, skillDomain, [graph.height - graph.bottom, graph.top]),
    };
  }

  function drawAxes(graph) {
    const plotRight = graph.width - graph.right;
    const plotBottom = graph.height - graph.bottom;
    const ink = cssVar("--figure-ink");
    const serif = cssVar("--research-serif");

    context.save();
    context.setLineDash([]);

    const zeroX = scale(0, biasDomain, [graph.left, plotRight]);
    const zeroY = scale(0, skillDomain, [plotBottom, graph.top]);
    context.strokeStyle = ink;
    context.globalAlpha = .46;
    context.lineWidth = 1.6;
    context.beginPath();
    context.moveTo(zeroX, graph.top);
    context.lineTo(zeroX, plotBottom);
    context.moveTo(graph.left, zeroY);
    context.lineTo(plotRight, zeroY);
    context.stroke();
    context.globalAlpha = 1;

    context.fillStyle = ink;
    context.font = `${graph.width < 600 ? 13 : 15}px ${serif}`;
    context.textAlign = "center";
    context.textBaseline = "top";
    axisTicks.forEach((bias) => {
      const x = scale(bias, biasDomain, [graph.left, plotRight]);
      context.fillText(bias === 0 ? "0" : bias.toFixed(1), x, plotBottom + 9);
    });

    context.textAlign = "right";
    context.textBaseline = "middle";
    let widestYTickWidth = 0;
    axisTicks.forEach((skill) => {
      const y = scale(skill, skillDomain, [plotBottom, graph.top]);
      const tickLabel = skill === 0 ? "0" : skill.toFixed(1);
      widestYTickWidth = Math.max(
        widestYTickWidth,
        context.measureText(tickLabel).width,
      );
      context.fillText(tickLabel, graph.left - 10, y);
    });

    const labelSize = graph.width < 600 ? 22 : 26;
    const labelHalfThickness = labelSize / 2;
    const minimumCanvasInset = labelHalfThickness + 2;
    const yTickLeft = graph.left - 10 - widestYTickWidth;
    const preferredTitleGap = graph.width < 600 ? 0 : 6;
    const yAxisTitleX = Math.max(
      minimumCanvasInset,
      yTickLeft - preferredTitleGap - labelHalfThickness,
    );
    const renderedTitleGap = Math.max(
      0,
      yTickLeft - (yAxisTitleX + labelHalfThickness),
    );
    stage.dataset.yAxisLabelGap = renderedTitleGap.toFixed(1);
    stage.dataset.yAxisLabelX = yAxisTitleX.toFixed(1);

    context.font = `700 ${labelSize}px ${serif}`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText("Bias, θ", (graph.left + plotRight) / 2, graph.height - 18);
    context.save();
    context.translate(yAxisTitleX, (graph.top + plotBottom) / 2);
    context.rotate(-Math.PI / 2);
    context.fillText("Skill, ℓ", 0, 0);
    context.restore();
    context.restore();
  }

  function drawCloud(graph, domain, progress) {
    const color = cssVar(domain.color);
    const radius = graph.width < 600 ? particleRadius.narrow : particleRadius.wide;

    context.fillStyle = color;
    context.globalAlpha = lerp(
      particleOpacity.prior,
      particleOpacity.posterior,
      progress,
    );
    context.beginPath();
    for (let index = 0; index < domain.prior.length; index += 1) {
      const prior = domain.prior[index];
      const posterior = domain.posterior[index];
      const point = graphPoint({
        skill: lerp(prior.skill, posterior.skill, progress),
        bias: lerp(prior.bias, posterior.bias, progress),
      }, graph);
      context.moveTo(point.x + radius, point.y);
      context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    }
    context.fill();

    const mean = graphPoint({
      skill: lerp(domain.priorStats.skill, domain.posteriorStats.skill, progress),
      bias: lerp(domain.priorStats.bias, domain.posteriorStats.bias, progress),
    }, graph);
    context.globalAlpha = .96;
    context.beginPath();
    context.arc(mean.x, mean.y, graph.width < 600 ? 3 : 4.2, 0, Math.PI * 2);
    context.fill();
    context.globalAlpha = 1;
  }

  function draw() {
    const graph = geometry();
    context.clearRect(0, 0, graph.width, graph.height);
    drawAxes(graph);
    for (const domain of domainClouds) drawCloud(graph, domain, visibleCollapse);
    drawParticleRadiusInset({
      context,
      graph,
      model: radiusModel,
      progress: visibleProgress,
      stage,
    });
    drawMeanRadiusSpiral({
      context,
      graph,
      model: spiralModel,
      progress: visibleProgress,
      stage,
    });
  }

  function syncCompletionState() {
    const isComplete = currentQuestionCount === totalQuestions
      && visibleProgress === 1
      && visibleCollapse === 1;
    stage.dataset.completionState = isComplete ? "complete" : "running";
  }

  function syncCounter() {
    const nextQuestionCount = Math.round(visibleProgress * totalQuestions);

    if (nextQuestionCount !== currentQuestionCount) {
      currentQuestionCount = nextQuestionCount;
      questionCount.textContent = String(currentQuestionCount);
    }

    questionCounter.setAttribute(
      "aria-label",
      `${currentQuestionCount} of ${totalQuestions} questions asked`,
    );
    stage.dataset.questionCount = String(currentQuestionCount);
  }

  function announceProgress() {
    const milestone = Math.round(visibleProgress * 4);
    const announcementKey = stage.dataset.completionState === "complete"
      ? "complete"
      : milestone;
    if (announcementKey === lastMilestone) return;
    lastMilestone = announcementKey;

    if (announcementKey === "complete") {
      status.textContent = `${totalQuestions} of ${totalQuestions} questions asked. All seven domain posteriors are resolved.`;
    } else if (milestone === 0) {
      status.textContent = "No questions asked. Seven broad domain priors are ready.";
    } else if (milestone === 4) {
      status.textContent = `${currentQuestionCount} of ${totalQuestions} questions asked. The seven domain posteriors are settling into their final state.`;
    } else {
      status.textContent = `${currentQuestionCount} of ${totalQuestions} questions asked. Seven domain posteriors are ${milestone * 25} percent through the scroll-driven update.`;
    }
  }

  function renderVisibleState() {
    stage.dataset.visibleProgress = visibleProgress.toFixed(3);
    stage.dataset.collapse = visibleCollapse.toFixed(3);
    syncCounter();
    syncCompletionState();
    announceProgress();
    draw();
  }

  function animateCollapse() {
    animationFrame = 0;
    const progressDifference = targetProgress - visibleProgress;
    const collapseDifference = targetCollapse - visibleCollapse;

    if (
      Math.abs(progressDifference) < .0004
      && Math.abs(collapseDifference) < .0005
    ) {
      visibleProgress = targetProgress;
      visibleCollapse = targetCollapse;
      renderVisibleState();
      return;
    }

    visibleProgress += progressDifference * questionProgressEase;
    visibleCollapse += collapseDifference * particleCollapseEase;
    renderVisibleState();
    animationFrame = requestAnimationFrame(animateCollapse);
  }

  function setScrollProgress(progress) {
    const bounded = Math.min(1, Math.max(0, progress));
    targetProgress = reducedMotion.matches ? 1 : bounded;
    targetCollapse = reducedMotion.matches ? 1 : slowCollapseProgress(bounded);
    stage.dataset.scrollProgress = bounded.toFixed(3);

    if (reducedMotion.matches) {
      visibleProgress = 1;
      targetCollapse = 1;
      visibleCollapse = 1;
      if (animationFrame) cancelAnimationFrame(animationFrame);
      animationFrame = 0;
      renderVisibleState();
      return;
    }

    syncCompletionState();
    if (!animationFrame) animationFrame = requestAnimationFrame(animateCollapse);
  }

  reducedMotion.addEventListener("change", () => {
    if (!reducedMotion.matches) return;
    targetProgress = 1;
    visibleProgress = 1;
    targetCollapse = 1;
    visibleCollapse = 1;
    if (animationFrame) cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    renderVisibleState();
  });

  createMarkerController({
    stage,
    track,
    steps: totalQuestions + 1,
    rangeDvh: 130,
    reducedMotion,
    onProgress: setScrollProgress,
  });

  const canvasObserver = observeCanvas(canvas, context, draw);
  setScrollProgress(reducedMotion.matches ? 1 : 0);

  return {
    redraw: canvasObserver.redraw,
  };
}
