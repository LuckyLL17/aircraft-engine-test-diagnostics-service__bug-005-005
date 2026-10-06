import test from 'node:test';
import assert from 'node:assert/strict';
import { diagnoseVibration } from '../src/domain/vibration/fuse.ts';
import { estimateRemainingLife } from '../src/domain/life/estimate.ts';
import { evaluatePerformance } from '../src/domain/performance/evaluate.ts';
import { replayTelemetry } from '../src/domain/timeline/replay.ts';
import { DomainError } from '../src/domain/errors.ts';

test('replay is deterministic and removes duplicate event ids', () => {
  const events = [
    { runId: 'run-1', eventId: 'b', sampleTimeMs: 2, receivedTimeMs: 4, sequence: 2, sensor: 'rpm', value: 900, unit: 'rpm' },
    { runId: 'run-1', eventId: 'a', sampleTimeMs: 1, receivedTimeMs: 3, sequence: 1, sensor: 'rpm', value: 800, unit: 'rpm' },
    { runId: 'run-1', eventId: 'b', sampleTimeMs: 2, receivedTimeMs: 4, sequence: 2, sensor: 'rpm', value: 900, unit: 'rpm' },
  ] as any;
  assert.deepEqual(replayTelemetry(events).map((item) => item.eventId), ['a', 'b']);
});

test('performance produces physical normalized indicators', () => {
  const result = evaluatePerformance({ rpm: 9_000, fuelFlowKgPerSecond: 2, compressorOutletPressureKpa: 900, turbineInletTemperatureK: 1_000, ambientPressureKpa: 101.3, ambientTemperatureK: 288, inletPressureKpa: 100, referenceRpm: 10_000, referenceTemperatureK: 288, referencePressureKpa: 101.3 });
  assert.ok(result.pressureRatio > 1);
  assert.ok(result.compressorEfficiency >= 0 && result.compressorEfficiency <= 1);
});

test('vibration evidence is grouped by correlated source', () => {
  const result = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 },
    { evidenceId: 'b', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 20, windowEndMs: 80, feature: 'broadband', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 },
    { evidenceId: 'c', sensorId: 's2', sourceGroup: 'g2', windowStartMs: 200, windowEndMs: 300, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 3, amplitude: 2 },
  ]);
  assert.equal(result[0].evidenceIds.length, 2);
});

test('overlapping chain from same sensor and source group is collapsed transitively', () => {
  const result = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 60, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 },
    { evidenceId: 'b', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 40, windowEndMs: 120, feature: 'broadband', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 },
    { evidenceId: 'c', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 100, windowEndMs: 180, feature: 'rotor-order', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 },
  ]);
  assert.deepEqual(result[0].evidenceIds, ['a']);
  // 代表窗只覆盖最终保留的证据，边界必能经 evidenceIds 追溯到原始输入。
  assert.deepEqual(result[0].representativeWindow, { startMs: 0, endMs: 60 });
  // 三条窗口本质上是同一份观测，后验只能按一条证据（ln4）计入，而不是 ln64。
  const expected = 0.05 * 4 / (1 - 0.05 + 0.05 * 4);
  assert.ok(Math.abs(result[0].posterior - expected) < 1e-12);
});

test('separated windows on the same sensor and group stay independent', () => {
  const result = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 },
    { evidenceId: 'b', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 200, windowEndMs: 300, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 },
  ]);
  assert.deepEqual(result[0].evidenceIds, ['a', 'b']);
});

test('overlapping windows from different sensors are independent even in the same group', () => {
  const result = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 },
    { evidenceId: 'b', sensorId: 's2', sourceGroup: 'g1', windowStartMs: 20, windowEndMs: 120, feature: 'casing-acceleration', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 },
  ]);
  assert.deepEqual(result[0].evidenceIds, ['a', 'b']);
});

test('representative prefers normal quality then larger unified weight', () => {
  const result = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'broadband', fault: 'bearing', likelihoodRatio: 9, amplitude: 4, quality: 'suspect' },
    { evidenceId: 'b', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 9, amplitude: 2 },
  ]);
  assert.deepEqual(result[0].evidenceIds, ['b']);

  const stronger = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'broadband', fault: 'bearing', likelihoodRatio: 9, amplitude: 2 },
    { evidenceId: 'b', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 9, amplitude: 4 },
  ])[0];
  assert.deepEqual(stronger.evidenceIds, ['b']);
});

test('location fusion uses inverse variance weighted by the same quality and amplitude rule', () => {
  const result = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 4, amplitude: 2, locationM: 1, locationUncertaintyM: 1 },
    { evidenceId: 'b', sensorId: 's2', sourceGroup: 'g2', windowStartMs: 200, windowEndMs: 300, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 4, amplitude: 2, locationM: 2, locationUncertaintyM: 1, quality: 'suspect' },
  ])[0];
  // 权重分别为 1.0 与 0.5，逆方差融合：位置 4/3，不确定度 sqrt(2/3)。
  assert.ok(Math.abs(result.locationM! - 4 / 3) < 1e-12);
  assert.ok(Math.abs(result.locationUncertaintyM! - Math.sqrt(2 / 3)) < 1e-12);
});

test('posterior is driven exactly by the retained evidence set', () => {
  const base = { sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope' as const, fault: 'bearing' as const, likelihoodRatio: 3, amplitude: 2 };
  const single = diagnoseVibration([{ ...base, evidenceId: 'a' }])[0];
  const duplicated = diagnoseVibration([
    { ...base, evidenceId: 'a' },
    { ...base, evidenceId: 'b', feature: 'broadband' },
    { ...base, evidenceId: 'c', feature: 'rotor-order' },
  ])[0];
  assert.equal(duplicated.evidenceIds.length, 1);
  assert.ok(Math.abs(single.posterior - duplicated.posterior) < 1e-15);
  const expected = (0.05 * 3) / (0.95 + 0.05 * 3);
  assert.ok(Math.abs(single.posterior - expected) < 1e-12);
});

test('suspect evidence discounts the log likelihood contribution through the shared weight', () => {
  const normal = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 9, amplitude: 4 },
  ])[0];
  const suspect = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 9, amplitude: 4, quality: 'suspect' },
  ])[0];
  // 正常：ln9*2；可疑：ln9；两者后验不同且可疑者更低。
  const expectedNormal = 1 / (1 + Math.exp(-(Math.log(0.05 / 0.95) + 2 * Math.log(9))));
  const expectedSuspect = 1 / (1 + Math.exp(-(Math.log(0.05 / 0.95) + Math.log(9))));
  assert.ok(Math.abs(normal.posterior - expectedNormal) < 1e-12);
  assert.ok(Math.abs(suspect.posterior - expectedSuspect) < 1e-12);
  assert.ok(suspect.posterior < normal.posterior);
});

test('location requires a paired positive uncertainty', () => {
  assert.throws(() => diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 4, amplitude: 2, locationM: 1 },
  ]), (error: unknown) => error instanceof DomainError && error.code === 'INVALID_LOCATION');
});

test('life estimate is reproducible for a fixed seed', () => {
  const input = { initialHealth: { compressorEfficiency: 0.95, turbineFlowCapacity: 0.96, bearingFriction: 0.02, hotSectionMargin: 0.9, elapsedHours: 100, version: 1 }, horizonHours: 10_000, samples: 100, seed: 42, meanDegradationPerHour: 0.0002, deviation: 0.00003, temperatureStress: 0.2, vibrationStress: 0.1, failureThreshold: 0.4, correlation: [[1, 0.3], [0.3, 1]] as [[number, number], [number, number]] };
  assert.deepEqual(estimateRemainingLife(input), estimateRemainingLife(input));
});
