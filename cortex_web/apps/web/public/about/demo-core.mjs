export const LAPSE = 0.025;
export const TEST_PARTICLE_COUNT = 520;

const TAU = Math.PI * 2;

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// The first half of the scroll runway intentionally produces less than one
// third of the visible contraction. This keeps the seven posterior clouds
// readable while they separate instead of resolving in one abrupt movement.
export function slowCollapseProgress(scrollProgress) {
  const normalized = clamp((scrollProgress - 0.06) / 0.94, 0, 1);
  return normalized ** 1.65;
}

// Abramowitz-Stegun approximation. More than sufficient for a visual demo and
// deterministic in browsers and Node.
export function erf(value) {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t)
    + 1.421413741) * t - 0.284496736) * t + 0.254829592)
    * t * Math.exp(-x * x);
  return sign * y;
}

export function normalCdf(value) {
  return 0.5 * (1 + erf(value / Math.SQRT2));
}

export function probitProbability(skill, bias, signal) {
  const z = Math.exp(skill) * (signal + bias);
  return LAPSE + (1 - 2 * LAPSE) * normalCdf(z);
}

export function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let x = state;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  const u = Math.max(1e-12, random());
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
}

export function createParticles(count = TEST_PARTICLE_COUNT, seed = 1701) {
  const random = mulberry32(seed);
  return Array.from({ length: count }, (_, index) => ({
    id: index,
    skill: clamp(0.52 + gaussian(random) * 0.68, -0.9, 2.4),
    bias: clamp(gaussian(random) * 0.82, -1.9, 1.9),
    weight: 1 / count,
  }));
}

export function weightedStats(particles) {
  let sumW = 0;
  let skill = 0;
  let bias = 0;
  let sumW2 = 0;
  for (const particle of particles) {
    sumW += particle.weight;
    skill += particle.weight * particle.skill;
    bias += particle.weight * particle.bias;
    sumW2 += particle.weight ** 2;
  }
  skill /= sumW;
  bias /= sumW;
  let skillVariance = 0;
  let biasVariance = 0;
  let covariance = 0;
  for (const particle of particles) {
    const w = particle.weight / sumW;
    const ds = particle.skill - skill;
    const db = particle.bias - bias;
    skillVariance += w * ds * ds;
    biasVariance += w * db * db;
    covariance += w * ds * db;
  }
  return {
    skill,
    bias,
    skillSd: Math.sqrt(Math.max(0, skillVariance)),
    biasSd: Math.sqrt(Math.max(0, biasVariance)),
    covariance,
    determinant: Math.max(0, skillVariance * biasVariance - covariance ** 2),
    ess: 1 / Math.max(1e-12, sumW2 / (sumW * sumW)),
  };
}

export function interpolatedParticleRadius(prior, posterior, progress) {
  if (prior.length !== posterior.length || prior.length === 0) {
    throw new RangeError("Particle clouds must be non-empty and have matching lengths.");
  }

  const bounded = clamp(progress, 0, 1);
  let totalWeight = 0;
  let meanSkill = 0;
  let meanBias = 0;

  for (let index = 0; index < prior.length; index += 1) {
    const priorParticle = prior[index];
    const posteriorParticle = posterior[index];
    const weight = priorParticle.weight
      + (posteriorParticle.weight - priorParticle.weight) * bounded;
    const skill = priorParticle.skill
      + (posteriorParticle.skill - priorParticle.skill) * bounded;
    const bias = priorParticle.bias
      + (posteriorParticle.bias - priorParticle.bias) * bounded;
    totalWeight += weight;
    meanSkill += weight * skill;
    meanBias += weight * bias;
  }

  meanSkill /= totalWeight;
  meanBias /= totalWeight;
  let squaredRadius = 0;

  for (let index = 0; index < prior.length; index += 1) {
    const priorParticle = prior[index];
    const posteriorParticle = posterior[index];
    const weight = (
      priorParticle.weight
      + (posteriorParticle.weight - priorParticle.weight) * bounded
    ) / totalWeight;
    const skill = priorParticle.skill
      + (posteriorParticle.skill - priorParticle.skill) * bounded;
    const bias = priorParticle.bias
      + (posteriorParticle.bias - priorParticle.bias) * bounded;
    squaredRadius += weight * (
      (skill - meanSkill) ** 2
      + (bias - meanBias) ** 2
    );
  }

  return Math.sqrt(Math.max(0, squaredRadius));
}

export function createParticleRadiusInterpolator(prior, posterior) {
  if (prior.length !== posterior.length || prior.length === 0) {
    throw new RangeError("Particle clouds must be non-empty and have matching lengths.");
  }

  const weightsMatch = prior.every((particle, index) => (
    Math.abs(particle.weight - posterior[index].weight) < 1e-12
  ));
  if (!weightsMatch) {
    return (progress) => interpolatedParticleRadius(prior, posterior, progress);
  }

  const priorStats = weightedStats(prior);
  const posteriorStats = weightedStats(posterior);
  const totalWeight = prior.reduce((sum, particle) => sum + particle.weight, 0);
  const priorSquaredRadius = priorStats.skillSd ** 2 + priorStats.biasSd ** 2;
  const posteriorSquaredRadius = (
    posteriorStats.skillSd ** 2 + posteriorStats.biasSd ** 2
  );
  let crossCovariance = 0;

  for (let index = 0; index < prior.length; index += 1) {
    const weight = prior[index].weight / totalWeight;
    crossCovariance += weight * (
      (prior[index].skill - priorStats.skill)
        * (posterior[index].skill - posteriorStats.skill)
      + (prior[index].bias - priorStats.bias)
        * (posterior[index].bias - posteriorStats.bias)
    );
  }

  return (progress) => {
    const bounded = clamp(progress, 0, 1);
    const remaining = 1 - bounded;
    const squaredRadius = (
      remaining ** 2 * priorSquaredRadius
      + bounded ** 2 * posteriorSquaredRadius
      + 2 * remaining * bounded * crossCovariance
    );
    return Math.sqrt(Math.max(0, squaredRadius));
  };
}

export function recenterParticles(particles, center = { skill: 0, bias: 0 }) {
  const stats = weightedStats(particles);
  return particles.map((particle) => ({
    ...particle,
    skill: particle.skill - stats.skill + center.skill,
    bias: particle.bias - stats.bias + center.bias,
  }));
}

function systematicResample(particles, random) {
  const count = particles.length;
  const cumulative = new Float64Array(count);
  let total = 0;
  for (let index = 0; index < count; index += 1) {
    total += particles[index].weight;
    cumulative[index] = total;
  }
  const output = [];
  let sourceIndex = 0;
  let cursor = random() / count;
  for (let index = 0; index < count; index += 1) {
    while (sourceIndex < count - 1 && cursor > cumulative[sourceIndex]) sourceIndex += 1;
    output.push(particles[sourceIndex]);
    cursor += 1 / count;
  }
  return output;
}

export function updateTestParticles(particles, target, questionIndex) {
  const count = particles.length;
  const contraction = Math.pow(0.82, Math.max(0, questionIndex - 1));
  const kernelSkill = Math.max(0.11, 0.62 * contraction);
  const kernelBias = Math.max(0.10, 0.72 * contraction);
  const side = questionIndex % 2 === 0 ? 1 : -1;
  const signal = clamp(-target.bias + side * 0.88 * Math.exp(-target.skill), -2.5, 2.5);
  const targetZ = Math.exp(target.skill) * (signal + target.bias);
  const response = targetZ >= 0 ? 1 : 0;
  const logWeights = new Float64Array(count);
  let maximum = -Infinity;

  for (let index = 0; index < count; index += 1) {
    const particle = particles[index];
    const ds = (particle.skill - target.skill) / kernelSkill;
    const db = (particle.bias - target.bias) / kernelBias;
    const probability = probitProbability(particle.skill, particle.bias, signal);
    const bernoulli = response === 1 ? probability : 1 - probability;
    // The bivariate kernel is deliberately pedagogical: it lets the pointer
    // represent a hypothetical learner state while the Bernoulli factor shows
    // the actual probit response channel.
    const logWeight = Math.log(Math.max(1e-12, particle.weight))
      - 0.5 * (ds * ds + db * db)
      + Math.log(Math.max(1e-9, bernoulli));
    logWeights[index] = logWeight;
    maximum = Math.max(maximum, logWeight);
  }

  let total = 0;
  const weighted = particles.map((particle, index) => {
    const weight = Math.exp(logWeights[index] - maximum);
    total += weight;
    return { ...particle, weight };
  });
  for (const particle of weighted) particle.weight /= total;
  const essBeforeResample = weightedStats(weighted).ess;

  const seed = (
    0x9e3779b9
    ^ Math.round((target.skill + 3) * 10007)
    ^ Math.round((target.bias + 3) * 7919)
    ^ (questionIndex * 2654435761)
  ) >>> 0;
  const random = mulberry32(seed);
  const parents = systematicResample(weighted, random);
  const attraction = Math.min(0.48, 0.25 + questionIndex * 0.025);
  const jitterSkill = Math.max(0.025, 0.15 * contraction);
  const jitterBias = Math.max(0.022, 0.17 * contraction);
  const next = parents.map((parent, index) => ({
    id: index,
    skill: clamp(
      parent.skill + attraction * (target.skill - parent.skill) + gaussian(random) * jitterSkill,
      -0.9,
      2.4,
    ),
    bias: clamp(
      parent.bias + attraction * (target.bias - parent.bias) + gaussian(random) * jitterBias,
      -1.9,
      1.9,
    ),
    weight: 1 / count,
  }));

  const stats = weightedStats(next);
  const probability = probitProbability(stats.skill, stats.bias, signal);
  return {
    particles: next,
    event: {
      questionIndex,
      signal,
      response,
      probability,
      z: Math.exp(stats.skill) * (signal + stats.bias),
      essBeforeResample,
      stats,
    },
  };
}

export function createLearningState(testStats) {
  const skill = clamp(testStats.skill, -0.5, 1.9);
  const bias = clamp(testStats.bias, -1.4, 1.4);
  return {
    trial: 0,
    skill,
    bias,
    skillCeiling: 2.25,
    alphaSkill: 0.13,
    alphaBias: 0.20,
    history: [{ trial: 0, skill, bias }],
    last: null,
  };
}

export function learningWeight(zAbs) {
  return zAbs * Math.exp((1 - zAbs * zAbs) / 2);
}

export function advanceLearning(state) {
  const side = Math.abs(state.bias) < 0.02
    ? (state.trial % 2 === 0 ? 1 : -1)
    : (state.bias > 0 ? -1 : 1);
  const sigma = Math.exp(-state.skill);
  const signal = side * 0.92 * sigma - state.bias;
  const gold = side > 0 ? 1 : 0;
  const z = Math.exp(state.skill) * (signal + state.bias);
  const probability = LAPSE + (1 - 2 * LAPSE) * normalCdf(z);
  const delta = probability - gold;
  const weight = learningWeight(Math.abs(z));
  const nextSkill = state.skill
    + state.alphaSkill * weight * (state.skillCeiling - state.skill);
  const nextBias = state.bias - state.alphaBias * delta;
  const trial = state.trial + 1;
  const last = {
    trial,
    signal,
    gold,
    z,
    probability,
    delta,
    weight,
    skillBefore: state.skill,
    biasBefore: state.bias,
    skillAfter: nextSkill,
    biasAfter: nextBias,
  };
  return {
    ...state,
    trial,
    skill: nextSkill,
    bias: nextBias,
    history: [...state.history, { trial, skill: nextSkill, bias: nextBias }],
    last,
  };
}

export function resetLearning(testStats) {
  return createLearningState(testStats);
}
