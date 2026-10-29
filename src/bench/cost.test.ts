import { describe, expect, it } from 'vitest';
import { median, percentile, segmentOffsets, summarise } from './cost';

describe('median', () => {
  it('handles odd, even and empty input without mutating it', () => {
    const values = [5, 1, 3];
    expect(median(values)).toBe(3);
    expect(values).toEqual([5, 1, 3]);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([])).toBeNaN();
  });
});

describe('percentile', () => {
  it('picks the nearest-rank value', () => {
    const values = Array.from({ length: 20 }, (_, i) => 20 - i); // 20 down to 1
    expect(percentile(values, 50)).toBe(10);
    expect(percentile(values, 95)).toBe(19);
    expect(percentile(values, 100)).toBe(20);
    expect(percentile([7], 95)).toBe(7);
    expect(percentile([], 50)).toBeNaN();
  });
});

describe('segmentOffsets', () => {
  it('spreads segments from the start to the end of the clip', () => {
    expect(segmentOffsets(100, 20, 3)).toEqual([0, 40, 80]);
    expect(segmentOffsets(100, 20, 1)).toEqual([0]);
  });

  it('returns one segment when it fills the clip, none when it is longer', () => {
    expect(segmentOffsets(100, 100, 3)).toEqual([0]);
    expect(segmentOffsets(100, 101, 3)).toEqual([]);
  });
});

describe('summarise', () => {
  it('reports medians, per-step decoder time and call RTF', () => {
    const row = summarise('base webgpu fp16/fp16', 5, [
      { featuresMs: 10, encoderMs: 400, decoderMs: 300, decoderSteps: 20, text: 'a' },
      { featuresMs: 12, encoderMs: 420, decoderMs: 600, decoderSteps: 40, text: 'b' },
      { featuresMs: 11, encoderMs: 410, decoderMs: 150, decoderSteps: 10, text: 'c' },
    ]);
    expect(row).toMatchObject({
      config: 'base webgpu fp16/fp16',
      seconds: 5,
      featuresMs: 11,
      encoderMs: 410,
      decoderMs: 300,
      decoderSteps: 20,
      msPerStep: 15,
      totalMs: 710,
      calls: 3,
      texts: ['a', 'b', 'c'],
    });
    expect(row.callRtf).toBeCloseTo(0.142);
  });

  it('leaves per-step time undefined when nothing was decoded', () => {
    expect(
      summarise('x', 2, [{ featuresMs: 1, encoderMs: 1, decoderMs: 0, decoderSteps: 0, text: '' }])
        .msPerStep,
    ).toBeNaN();
  });
});
