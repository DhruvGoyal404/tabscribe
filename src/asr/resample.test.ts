import { describe, expect, it } from 'vitest';
import { ASR_SAMPLE_RATE, downmixToMono, Resampler, resampleTo16k } from './resample';

const sine = (frequency: number, rate: number, seconds: number, amplitude = 1): Float32Array =>
  Float32Array.from(
    { length: Math.round(rate * seconds) },
    (_, i) => amplitude * Math.sin((2 * Math.PI * frequency * i) / rate),
  );

/** RMS over the middle of a signal, away from the start/end transients. */
function middleRms(signal: Float32Array, trim = 0.1): number {
  const start = Math.floor(signal.length * trim);
  const end = Math.ceil(signal.length * (1 - trim));
  let sum = 0;
  for (let i = start; i < end; i++) sum += (signal[i] ?? 0) ** 2;
  return Math.sqrt(sum / (end - start));
}

/** Amplitude of one frequency component (single-bin DFT over the middle of the signal). */
function toneAmplitude(signal: Float32Array, frequency: number, rate: number): number {
  const start = Math.floor(signal.length * 0.1);
  const end = Math.ceil(signal.length * 0.9);
  let re = 0;
  let im = 0;
  for (let i = start; i < end; i++) {
    const phase = (2 * Math.PI * frequency * i) / rate;
    re += (signal[i] ?? 0) * Math.cos(phase);
    im -= (signal[i] ?? 0) * Math.sin(phase);
  }
  return (2 * Math.hypot(re, im)) / (end - start);
}

const db = (ratio: number): number => 20 * Math.log10(ratio);

describe('Resampler length', () => {
  it.each([48_000, 44_100, 32_000, 22_050, 96_000, 16_000, 8_000])(
    '%i Hz: one second becomes 16 000 samples',
    (rate) => {
      expect(resampleTo16k([new Float32Array(rate)], rate)).toHaveLength(ASR_SAMPLE_RATE);
    },
  );

  it('matches ceil(n * 16000 / rate) for odd lengths', () => {
    for (const [rate, n] of [
      [48_000, 1],
      [48_000, 2],
      [48_000, 4],
      [44_100, 1001],
      [44_100, 12_345],
    ] as const) {
      expect(resampleTo16k([new Float32Array(n)], rate)).toHaveLength(
        Math.ceil((n * ASR_SAMPLE_RATE) / rate),
      );
    }
  });

  it('returns nothing for empty input', () => {
    expect(resampleTo16k([new Float32Array(0)], 48_000)).toHaveLength(0);
    expect(resampleTo16k([], 48_000)).toHaveLength(0);
  });

  it('rejects invalid rates', () => {
    expect(() => new Resampler({ inputRate: 0 })).toThrow(RangeError);
    expect(() => new Resampler({ inputRate: 44_100.5 })).toThrow(RangeError);
  });
});

describe('Resampler frequency response', () => {
  it.each([48_000, 44_100])('%i Hz: DC passes at unity gain', (rate) => {
    const out = resampleTo16k([new Float32Array(rate).fill(0.5)], rate);
    expect(middleRms(out)).toBeCloseTo(0.5, 4);
  });

  it.each([
    [48_000, 440],
    [48_000, 1_000],
    [48_000, 3_000],
    [48_000, 6_000],
    [44_100, 1_000],
    [44_100, 6_000],
  ])('%i Hz: a %i Hz tone passes within 0.1 dB', (rate, frequency) => {
    const out = resampleTo16k([sine(frequency, rate, 1)], rate);
    expect(Math.abs(db(toneAmplitude(out, frequency, ASR_SAMPLE_RATE)))).toBeLessThan(0.1);
  });

  it.each([
    // Input tone, the frequency it would alias to at 16 kHz without a low-pass.
    [48_000, 10_000, 6_000],
    [48_000, 12_000, 4_000],
    [48_000, 15_000, 1_000],
    [48_000, 20_000, 4_000],
    [44_100, 9_000, 7_000],
    [44_100, 13_000, 3_000],
  ])('%i Hz: a %i Hz tone is removed, not aliased to %i Hz', (rate, frequency, alias) => {
    const out = resampleTo16k([sine(frequency, rate, 1)], rate);
    expect(db(toneAmplitude(out, alias, ASR_SAMPLE_RATE))).toBeLessThan(-70);
    expect(db(middleRms(out))).toBeLessThan(-70);
  });

  it('upsamples 8 kHz without inventing content above the old Nyquist', () => {
    const out = resampleTo16k([sine(1_000, 8_000, 1)], 8_000);
    expect(Math.abs(db(toneAmplitude(out, 1_000, ASR_SAMPLE_RATE)))).toBeLessThan(0.1);
    // The mirror image of 1 kHz around 4 kHz.
    expect(db(toneAmplitude(out, 7_000, ASR_SAMPLE_RATE))).toBeLessThan(-70);
  });

  it('is the identity at 16 kHz', () => {
    const input = sine(1_000, ASR_SAMPLE_RATE, 0.1);
    expect(resampleTo16k([input], ASR_SAMPLE_RATE)).toEqual(input);
  });
});

describe('Resampler streaming', () => {
  it.each([48_000, 44_100])('%i Hz: output does not depend on chunk sizes', (rate) => {
    const input = sine(1_234, rate, 0.5, 0.8);
    const oneShot = resampleTo16k([input], rate);

    const resampler = new Resampler({ inputRate: rate });
    const parts: Float32Array[] = [];
    // 128-sample render quanta, plus odd sizes, to cover every boundary case.
    const sizes = [128, 1, 7, 128, 333, 2048, 5];
    for (let offset = 0, i = 0; offset < input.length; i++) {
      const size = sizes[i % sizes.length] ?? 128;
      parts.push(resampler.push(input.subarray(offset, offset + size)));
      offset += size;
    }
    parts.push(resampler.flush());
    const streamed = Float32Array.from(parts.flatMap((part) => [...part]));

    expect(streamed).toHaveLength(oneShot.length);
    for (let i = 0; i < oneShot.length; i++) {
      expect(streamed[i]).toBeCloseTo(oneShot[i] ?? NaN, 6);
    }
  });

  it('can be reused after flush', () => {
    const resampler = new Resampler({ inputRate: 48_000 });
    const input = sine(500, 48_000, 0.1);
    const first = [...resampler.push(input), ...resampler.flush()];
    const second = [...resampler.push(input), ...resampler.flush()];
    expect(second).toEqual(first);
  });

  it('has no delay: a centred impulse stays centred', () => {
    const input = new Float32Array(48_000);
    input[24_000] = 1;
    const out = resampleTo16k([input], 48_000);
    let peak = 0;
    for (let i = 1; i < out.length; i++) if ((out[i] ?? 0) > (out[peak] ?? 0)) peak = i;
    expect(peak).toBe(8_000);
  });
});

describe('downmixToMono', () => {
  it('averages channels', () => {
    const left = Float32Array.from([1, 0.5, -1]);
    const right = Float32Array.from([0, 0.5, 1]);
    expect([...downmixToMono([left, right])]).toEqual([0.5, 0.5, 0]);
  });

  it('cancels out-of-phase stereo and halves a one-sided signal', () => {
    const tone = sine(1_000, 48_000, 0.1);
    const inverted = tone.map((x) => -x);
    expect(middleRms(downmixToMono([tone, inverted]))).toBe(0);
    expect(middleRms(downmixToMono([tone, new Float32Array(tone.length)]))).toBeCloseTo(
      middleRms(tone) / 2,
      6,
    );
  });

  it('passes mono through and rejects mismatched channels', () => {
    const mono = Float32Array.from([0.1, 0.2]);
    expect(downmixToMono([mono])).toBe(mono);
    expect(downmixToMono([])).toHaveLength(0);
    expect(() => downmixToMono([new Float32Array(2), new Float32Array(3)])).toThrow(RangeError);
  });

  it('feeds the resampler: stereo 48 kHz becomes mono 16 kHz', () => {
    const tone = sine(1_000, 48_000, 1, 0.5);
    const out = resampleTo16k([tone, tone], 48_000);
    expect(out).toHaveLength(16_000);
    expect(Math.abs(db(toneAmplitude(out, 1_000, ASR_SAMPLE_RATE) / 0.5))).toBeLessThan(0.1);
  });
});
