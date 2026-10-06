import test from 'node:test';
import assert from 'node:assert/strict';
import { diagnoseVibration } from '../src/domain/vibration/fuse.ts';
import { estimateRemainingLife } from '../src/domain/life/estimate.ts';
import { evaluatePerformance } from '../src/domain/performance/evaluate.ts';
import { replayTelemetry } from '../src/domain/timeline/replay.ts';

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

test('vibration dedup keeps the stronger representative of correlated windows', () => {
  const result = diagnoseVibration([
    { evidenceId: 'weak', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'broadband', fault: 'bearing', likelihoodRatio: 2, amplitude: 1 },
    { evidenceId: 'strong', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 50, windowEndMs: 150, feature: 'broadband', fault: 'bearing', likelihoodRatio: 2, amplitude: 8 },
    { evidenceId: 'independent', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 200, windowEndMs: 300, feature: 'broadband', fault: 'bearing', likelihoodRatio: 2, amplitude: 1 },
  ]);
  assert.deepEqual(result[0].evidenceIds, ['strong', 'independent']);
  assert.deepEqual(result[0].representativeWindow, { startMs: 50, endMs: 300 });
});

test('vibration disconfirming evidence lowers confidence below the prior', () => {
  const result = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'broadband', fault: 'bearing', likelihoodRatio: 0.5, amplitude: 2 },
  ], 0.1);
  assert.ok(result[0].posterior < 0.1);
});

test('vibration location fusion shrinks uncertainty with independent sensors', () => {
  const result = diagnoseVibration([
    { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 2, amplitude: 2, locationM: 1, locationUncertaintyM: 0.2 },
    { evidenceId: 'b', sensorId: 's2', sourceGroup: 'g2', windowStartMs: 0, windowEndMs: 100, feature: 'bearing-envelope', fault: 'bearing', likelihoodRatio: 2, amplitude: 2, locationM: 1.4, locationUncertaintyM: 0.2 },
  ]);
  assert.equal(result[0].evidenceIds.length, 2);
  assert.ok(Math.abs(result[0].locationM! - 1.2) < 1e-9);
  assert.ok(Math.abs(result[0].locationUncertaintyM! - 0.2 / Math.SQRT2) < 1e-9);
});

test('vibration quality follows the same weight rule as amplitude', () => {
  const evidence = { evidenceId: 'a', sensorId: 's1', sourceGroup: 'g1', windowStartMs: 0, windowEndMs: 100, feature: 'broadband', fault: 'bearing', likelihoodRatio: 4, amplitude: 2 } as const;
  const normal = diagnoseVibration([{ ...evidence, quality: 'normal' }], 0.05)[0].posterior;
  const suspect = diagnoseVibration([{ ...evidence, quality: 'suspect' }], 0.05)[0].posterior;
  assert.ok(suspect < normal);
});

test('life estimate is reproducible for a fixed seed', () => {
  const input = { initialHealth: { compressorEfficiency: 0.95, turbineFlowCapacity: 0.96, bearingFriction: 0.02, hotSectionMargin: 0.9, elapsedHours: 100, version: 1 }, horizonHours: 10_000, samples: 100, seed: 42, meanDegradationPerHour: 0.0002, deviation: 0.00003, temperatureStress: 0.2, vibrationStress: 0.1, failureThreshold: 0.4, correlation: [[1, 0.3], [0.3, 1]] as [[number, number], [number, number]] };
  assert.deepEqual(estimateRemainingLife(input), estimateRemainingLife(input));
});
