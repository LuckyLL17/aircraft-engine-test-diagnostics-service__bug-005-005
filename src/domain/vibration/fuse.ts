import { invariant } from '../errors.ts';
import type { FaultCandidate, VibrationEvidence } from '../types.ts';

const SUSPECT_QUALITY_WEIGHT = 0.5;
const MIN_AMPLITUDE_WEIGHT = 0.25;
const MAX_AMPLITUDE_WEIGHT = 2;

/**
 * 统一证据权重：质量等级与振幅对所有下游量（对数似然比、后验概率、定位融合）
 * 使用同一套折扣规则，任何渠道都不得另设权重。
 */
export function evidenceWeight(item: VibrationEvidence): number {
  const qualityWeight = item.quality === 'suspect' ? SUSPECT_QUALITY_WEIGHT : 1;
  const amplitudeWeight = Math.max(MIN_AMPLITUDE_WEIGHT, Math.min(MAX_AMPLITUDE_WEIGHT, item.amplitude / 2));
  return qualityWeight * amplitudeWeight;
}

function overlaps(a: VibrationEvidence, b: VibrationEvidence): boolean {
  return Math.max(a.windowStartMs, b.windowStartMs) <= Math.min(a.windowEndMs, b.windowEndMs);
}

/**
 * 两条证据是否为同一物理观测的重复特征：
 * 同一传感器、同一源组（未标注源组视为同一未归属组）、时间窗重叠。
 * 特征类型不同不构成独立性；不同传感器、不同源组或互不重叠的窗口才是独立证据。
 */
function isDuplicateObservation(a: VibrationEvidence, b: VibrationEvidence): boolean {
  return (
    a.sensorId === b.sensorId
    && (a.sourceGroup ?? '') === (b.sourceGroup ?? '')
    && overlaps(a, b)
  );
}

/**
 * 在同一故障标签内按重复关系做连通片聚类（并查集），保证重叠关系可传递：
 * A∩B、B∩C 时即使 A 与 C 不直接重叠也合并为一份证据。
 */
function clusterRepresentatives(items: VibrationEvidence[]): VibrationEvidence[] {
  const parent = items.map((_, index) => index);
  const find = (index: number): number => {
    let root = index;
    while (parent[root] !== root) root = parent[root];
    while (parent[index] !== index) {
      const next = parent[index];
      parent[index] = root;
      index = next;
    }
    return root;
  };
  const union = (left: number, right: number): void => {
    parent[find(left)] = find(right);
  };

  for (let i = 0; i < items.length; i += 1) {
    for (let j = i + 1; j < items.length; j += 1) {
      if (isDuplicateObservation(items[i], items[j])) union(i, j);
    }
  }

  const clusters = new Map<number, VibrationEvidence[]>();
  items.forEach((item, index) => {
    const root = find(index);
    clusters.set(root, [...(clusters.get(root) ?? []), item]);
  });
  return [...clusters.values()].map(chooseRepresentative);
}

/**
 * 代表证据必须是原始输入之一，保证证据 ID 与时间窗可追溯：
 * 先取质量等级更高者，再取统一权重更大者，最后以起始时间与证据 ID 做确定性兜底。
 */
function chooseRepresentative(members: VibrationEvidence[]): VibrationEvidence {
  return members.reduce((best, item) => {
    if (best.quality !== 'suspect' && item.quality === 'suspect') return best;
    if (best.quality === 'suspect' && item.quality !== 'suspect') return item;
    const weightDifference = evidenceWeight(item) - evidenceWeight(best);
    if (weightDifference > 0) return item;
    if (weightDifference < 0) return best;
    if (item.windowStartMs !== best.windowStartMs) return item.windowStartMs < best.windowStartMs ? item : best;
    return item.evidenceId.localeCompare(best.evidenceId) < 0 ? item : best;
  });
}

function validate(item: VibrationEvidence): VibrationEvidence {
  invariant(item.windowEndMs > item.windowStartMs, 'INVALID_VIBRATION_WINDOW', 'Vibration window must have positive duration', { evidenceId: item.evidenceId });
  invariant(Number.isFinite(item.likelihoodRatio) && item.likelihoodRatio > 0, 'INVALID_LIKELIHOOD_RATIO', 'Likelihood ratio must be positive', { evidenceId: item.evidenceId });
  invariant(Number.isFinite(item.amplitude) && item.amplitude >= 0, 'INVALID_VIBRATION_AMPLITUDE', 'Vibration amplitude must be non-negative', { evidenceId: item.evidenceId });
  const hasLocation = item.locationM !== undefined;
  const hasUncertainty = item.locationUncertaintyM !== undefined;
  invariant(hasLocation === hasUncertainty, 'INVALID_LOCATION', 'Location and its uncertainty must be provided together', { evidenceId: item.evidenceId });
  if (hasLocation) {
    invariant(Number.isFinite(item.locationM) && Number.isFinite(item.locationUncertaintyM) && item.locationUncertaintyM! > 0, 'INVALID_LOCATION', 'Location uncertainty must be a positive finite number', { evidenceId: item.evidenceId });
  }
  return item;
}

export function diagnoseVibration(evidence: VibrationEvidence[], prior = 0.05): FaultCandidate[] {
  invariant(prior > 0 && prior < 1, 'INVALID_PRIOR', 'Fault prior must be between zero and one');
  const valid = evidence.map(validate);

  const byFault = new Map<string, VibrationEvidence[]>();
  for (const item of valid) {
    if (item.fault === 'unknown') continue;
    byFault.set(item.fault, [...(byFault.get(item.fault) ?? []), item]);
  }

  return [...byFault.entries()].map(([fault, items]) => {
    // 每个连通片只保留一份代表证据；独立窗口（异传感器/异源组/不重叠）各自保留。
    const representatives = clusterRepresentatives(items)
      .sort((a, b) => a.windowStartMs - b.windowStartMs || a.evidenceId.localeCompare(b.evidenceId));

    // 后验只由最终保留的证据驱动：每条证据恰好贡献一次 w·ln(LR)，
    // 证据总量（evidenceIds）与故障置信度由此严格对应。
    const score = representatives.reduce((sum, item) => sum + Math.log(item.likelihoodRatio) * evidenceWeight(item), 0);
    const priorLogOdds = Math.log(prior / (1 - prior));
    const clampedScore = Math.max(-500, Math.min(500, score));
    const posterior = score === 0 ? prior : 1 / (1 + Math.exp(-(priorLogOdds + clampedScore)));

    // 定位采用逆方差融合，并复用同一质量/振幅权重：有效精度 w_i/σ_i²。
    const located = representatives.filter((item) => item.locationM !== undefined);
    const precision = located.reduce((sum, item) => sum + evidenceWeight(item) / item.locationUncertaintyM! ** 2, 0);
    const locationM = precision
      ? located.reduce((sum, item) => sum + evidenceWeight(item) * item.locationM! / item.locationUncertaintyM! ** 2, 0) / precision
      : null;
    const locationUncertaintyM = precision ? 1 / Math.sqrt(precision) : null;

    return {
      fault: fault as FaultCandidate['fault'],
      posterior,
      evidenceIds: representatives.map((item) => item.evidenceId),
      // 代表窗取所有保留证据时间窗的并集包络，边界均可经 evidenceIds 追溯到原始输入。
      representativeWindow: {
        startMs: Math.min(...representatives.map((item) => item.windowStartMs)),
        endMs: Math.max(...representatives.map((item) => item.windowEndMs)),
      },
      locationM,
      locationUncertaintyM,
    };
  }).sort((a, b) => b.posterior - a.posterior || a.fault.localeCompare(b.fault));
}
