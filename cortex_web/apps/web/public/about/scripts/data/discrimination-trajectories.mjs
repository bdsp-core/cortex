import { eegDomains } from "./domains.mjs";

const TAU = Math.PI * 2;

export const cortexQuestionMaximum = 250;
export const startingAuroc = .75;

export const discriminationCohorts = Object.freeze([
  Object.freeze({
    key: "experts",
    label: "Experts",
    color: "--domain-spike",
    dash: Object.freeze([]),
    count: 16,
    stopRange: Object.freeze([72, 236]),
    targetSpread: .13,
    convergence: 2.75,
    noise: .03,
    seed: 1801,
  }),
  Object.freeze({
    key: "experienced",
    label: "Experienced",
    color: "--domain-lrda",
    dash: Object.freeze([7, 4]),
    count: 22,
    stopRange: Object.freeze([205, cortexQuestionMaximum]),
    targetSpread: .145,
    convergence: 1.08,
    noise: .043,
    seed: 2903,
  }),
  Object.freeze({
    key: "novices",
    label: "Novices",
    color: "--domain-gpd",
    dash: Object.freeze([2.5, 3.5]),
    count: 17,
    stopRange: Object.freeze([78, 248]),
    targetSpread: .135,
    convergence: 2.55,
    noise: .034,
    seed: 3911,
  }),
]);

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalizedExponential(progress, rate) {
  return (1 - Math.exp(-rate * progress)) / (1 - Math.exp(-rate));
}

function cohortTargetCenter(domain, cohortKey) {
  if (cohortKey === "experts") return domain.auroc.experts;
  if (cohortKey === "novices") return domain.auroc.nonexperts;
  return (
    domain.auroc.nonexperts
    + (domain.auroc.experts - domain.auroc.nonexperts) * .53
  );
}

function createReader(domain, cohort, readerIndex) {
  const random = mulberry32(
    domain.seed * 3
    + cohort.seed
    + readerIndex * 7919,
  );
  const centeredRank = (
    ((readerIndex + .5) / cohort.count) * 2 - 1
  );
  const distributionOffset = cohort.targetSpread * (
    centeredRank * .88
    + (random() * 2 - 1) * .12
  );
  const target = clamp(
    cohortTargetCenter(domain, cohort.key) + distributionOffset,
    .505,
    .995,
  );
  const stop = (
    cohort.key === "experienced" && readerIndex === cohort.count - 1
      ? cortexQuestionMaximum
      : Math.round(
        cohort.stopRange[0]
        + random() * (cohort.stopRange[1] - cohort.stopRange[0]),
      )
  );
  const phase = random() * TAU;
  const frequency = .075 + random() * .075;
  let deviation = 0;

  const points = Array.from({ length: stop + 1 }, (_, question) => {
    if (question === 0) {
      return { question, auroc: startingAuroc };
    }
    if (question === stop) {
      return { question, auroc: target };
    }

    const progress = question / stop;
    const convergence = cohort.key === "experienced"
      ? progress ** 1.42
      : normalizedExponential(progress, cohort.convergence);
    deviation = .83 * deviation + .17 * (random() * 2 - 1);
    const oscillation = (
      Math.sin(question * frequency + phase) * .55
      + deviation * .8
    );
    const noiseEnvelope = Math.sin(Math.PI * progress) ** .72;
    const auroc = (
      startingAuroc
      + (target - startingAuroc) * convergence
      + oscillation * cohort.noise * noiseEnvelope
    );

    return {
      question,
      auroc: clamp(auroc, .5, 1),
    };
  });

  return Object.freeze({
    id: `${domain.key}-${cohort.key}-${String(readerIndex + 1).padStart(2, "0")}`,
    domain: domain.key,
    cohort: cohort.key,
    color: cohort.color,
    dash: cohort.dash,
    stop,
    target,
    points: Object.freeze(points.map(Object.freeze)),
  });
}

export function createDiscriminationTrajectories(domain = eegDomains[0]) {
  return Object.freeze(discriminationCohorts.flatMap((cohort) => (
    Array.from(
      { length: cohort.count },
      (_, index) => createReader(domain, cohort, index),
    )
  )));
}

export function createDomainDiscriminationTrajectories() {
  return Object.freeze(eegDomains.map((domain) => Object.freeze({
    key: domain.key,
    label: domain.label,
    color: domain.color,
    trajectories: createDiscriminationTrajectories(domain),
  })));
}

export const readersPerDomain = discriminationCohorts.reduce(
  (total, cohort) => total + cohort.count,
  0,
);
export const domainDiscriminationTrajectories = (
  createDomainDiscriminationTrajectories()
);
export const discriminationTrajectories = Object.freeze(
  domainDiscriminationTrajectories.flatMap(({ trajectories }) => trajectories),
);

const experiencedTrajectories = discriminationTrajectories.filter(
  ({ cohort }) => cohort === "experienced",
);
const comparisonTrajectories = discriminationTrajectories.filter(
  ({ cohort }) => cohort !== "experienced",
);
const meanStop = (trajectories) => (
  trajectories.reduce((total, { stop }) => total + stop, 0)
  / trajectories.length
);

export const experiencedMeanStopRatio = (
  meanStop(experiencedTrajectories) / meanStop(comparisonTrajectories)
);
