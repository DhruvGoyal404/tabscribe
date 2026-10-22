import { describe, expect, it } from 'vitest';
import { blockSizeFor, downmix, LevelMeter, MIN_DBFS, toDbfs } from './level';

const SAMPLE_RATE = 48000;
const BLOCK = blockSizeFor(SAMPLE_RATE); // 2400 samples = 50 ms

function sine(amplitude: number, frequency: number, length: number): Float32Array {
  return Float32Array.from(
    { length },
    (_, i) => amplitude * Math.sin((2 * Math.PI * frequency * i) / SAMPLE_RATE),
  );
}

describe('toDbfs', () => {
  it('maps full scale to 0 dBFS and halves to about -6.02', () => {
    expect(toDbfs(1)).toBe(0);
    expect(toDbfs(0.5)).toBeCloseTo(-6.0206, 4);
  });

  it('floors silence and tiny values at MIN_DBFS', () => {
    expect(toDbfs(0)).toBe(MIN_DBFS);
    expect(toDbfs(1e-9)).toBe(MIN_DBFS);
  });
});

describe('blockSizeFor', () => {
  it('sizes 50 ms blocks for common rates', () => {
    expect(blockSizeFor(48000)).toBe(2400);
    expect(blockSizeFor(44100)).toBe(2205);
    expect(blockSizeFor(16000)).toBe(800);
  });
});

describe('LevelMeter', () => {
  it('reports silence at the floor without clipping', () => {
    expect(new LevelMeter(BLOCK).push(new Float32Array(BLOCK))).toEqual({
      rmsDbfs: MIN_DBFS,
      peakDbfs: MIN_DBFS,
      clipped: false,
    });
  });

  it('reads a full-scale sine at about -3.01 dBFS RMS and 0 dBFS peak', () => {
    // 1 kHz at 48 kHz: 48 samples per period, so a 2400-sample block holds whole periods.
    const level = new LevelMeter(BLOCK).push(sine(1, 1000, BLOCK));
    expect(level?.rmsDbfs).toBeCloseTo(-3.0103, 3);
    expect(level?.peakDbfs).toBeCloseTo(0, 6);
    expect(level?.clipped).toBe(false);
  });

  it('reads a half-scale sine about 6 dB lower', () => {
    expect(new LevelMeter(BLOCK).push(sine(0.5, 1000, BLOCK))?.rmsDbfs).toBeCloseTo(-9.0309, 3);
  });

  it('flags clipping when samples exceed full scale', () => {
    const square = Float32Array.from({ length: BLOCK }, (_, i) => (i % 2 === 0 ? 2 : -2));
    const level = new LevelMeter(BLOCK).push(square);
    expect(level?.rmsDbfs).toBeCloseTo(6.0206, 3);
    expect(level?.peakDbfs).toBeCloseTo(6.0206, 3);
    expect(level?.clipped).toBe(true);
  });

  it('accumulates 128-frame render quanta and emits once per block', () => {
    const meter = new LevelMeter(BLOCK);
    const signal = sine(1, 1000, BLOCK * 2);
    const emitted: number[] = [];
    for (let offset = 0; offset < signal.length; offset += 128) {
      const level = meter.push(signal.subarray(offset, offset + 128));
      if (level) emitted.push(offset + 128);
    }
    // 2400 isn't a multiple of 128, so each block completes inside a quantum.
    expect(emitted).toEqual([2432, 4864]);
  });

  it('resets between blocks', () => {
    const meter = new LevelMeter(4);
    expect(meter.push(new Float32Array([1, 1, 1, 1]))?.rmsDbfs).toBe(0);
    expect(meter.push(new Float32Array(4))?.rmsDbfs).toBe(MIN_DBFS);
  });

  it('treats NaN and Infinity as silence', () => {
    expect(new LevelMeter(4).push(new Float32Array([NaN, Infinity, -Infinity, 0]))).toEqual({
      rmsDbfs: MIN_DBFS,
      peakDbfs: MIN_DBFS,
      clipped: false,
    });
  });

  it.each([0, -1, 1.5])('rejects block size %s', (size) => {
    expect(() => new LevelMeter(size)).toThrow(RangeError);
  });
});

describe('downmix', () => {
  it('averages stereo channels', () => {
    const out = downmix(
      [new Float32Array([1, 0.5]), new Float32Array([0, -0.5])],
      new Float32Array(2),
    );
    expect(Array.from(out)).toEqual([0.5, 0]);
  });

  it('copies a single channel unchanged', () => {
    expect(Array.from(downmix([new Float32Array([0.25, -1])], new Float32Array(2)))).toEqual([
      0.25, -1,
    ]);
  });

  it('cancels out-of-phase stereo to silence', () => {
    const left = sine(1, 1000, 256);
    const right = left.map((s) => -s);
    expect(downmix([left, right], new Float32Array(256)).every((s) => s === 0)).toBe(true);
  });

  it('returns zeros when there are no channels', () => {
    expect(Array.from(downmix([], new Float32Array([9, 9])))).toEqual([0, 0]);
  });
});
