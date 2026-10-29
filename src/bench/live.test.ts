import { describe, expect, it } from 'vitest';
import type { DecodeJob, Decoded } from '../asr/scheduler';
import { LatencyLog } from './live';

const job = (endSample: number): DecodeJob => ({
  id: 0,
  kind: 'final',
  audio: new Float32Array(1),
  prompt: null,
  segmentIds: [0],
  startSample: 0,
  endSample,
  reason: 'silence',
  overlapSamples: 0,
});

const result = {} as Decoded;

describe('LatencyLog', () => {
  it('times a final from where its speech ended, and a partial from its audio end', () => {
    const log = new LatencyLog(1_000);
    log.segment({
      type: 'final',
      segmentId: 4,
      startSample: 0,
      endSample: 51_200, // 3.2 s, padded past the end of speech
      speechEndSample: 48_000, // 3 s
      reason: 'silence',
      overlapSamples: 0,
      audio: new Float32Array(51_200),
    });
    // Speech ended at 1000 + 3000 ms; the final showed at 5200 ms.
    log.shown(
      { type: 'final', segmentIds: [3, 4], startSample: 0, endSample: 51_200, text: 'hi', result },
      job(51_200),
      5_200,
    );
    // The partial previewed audio up to 2 s (3000 ms on the clock) and showed at 3400 ms.
    log.shown({ type: 'partial', segmentId: 5, text: 'he' }, job(32_000), 3_400);
    expect(log.segments).toBe(1);
    expect(log.finalMs).toEqual([1_200]);
    expect(log.partialMs).toEqual([400]);
  });

  it('skips captions without text, and finals it saw no segment for', () => {
    const log = new LatencyLog(0);
    log.shown({ type: 'partial', segmentId: 0, text: '' }, job(16_000), 2_000);
    log.shown(
      { type: 'final', segmentIds: [9], startSample: 0, endSample: 16_000, text: 'x', result },
      job(16_000),
      2_000,
    );
    expect(log.partialMs).toEqual([]);
    expect(log.finalMs).toEqual([]);
  });
});
