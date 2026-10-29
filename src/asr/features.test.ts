import { WhisperFeatureExtractor } from '@huggingface/transformers';
import { describe, expect, it } from 'vitest';
import {
  audioLength,
  padFrames,
  skipPaddingFrames,
  trimmedLength,
  type WhisperFeatureExtractorLike,
} from './features';

// openai/whisper-base's preprocessor_config.json; the mel filters are computed from it.
const CONFIG = {
  feature_extractor_type: 'WhisperFeatureExtractor',
  feature_size: 80,
  hop_length: 160,
  n_fft: 400,
  n_samples: 480_000,
  nb_max_frames: 3000,
  sampling_rate: 16_000,
};
const GEOMETRY = { hop: 160, frameLength: 400, maxFrames: 3000 };

type Extractor = WhisperFeatureExtractorLike &
  ((audio: Float32Array) => Promise<{ input_features: { data: Float32Array; dims: number[] } }>);
const extractor = (): Extractor => new WhisperFeatureExtractor(CONFIG);

/** Speech-like test audio: a few partials with a slow envelope and a little noise. */
function audio(seconds: number, seed = 1): Float32Array {
  let state = seed;
  const noise = () => ((state = (state * 1_103_515_245 + 12_345) % 2 ** 31) / 2 ** 31 - 0.5) * 0.02;
  return Float32Array.from({ length: Math.round(seconds * 16_000) }, (_, i) => {
    const t = i / 16_000;
    const envelope = 0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * t);
    return (
      envelope * 0.3 * (Math.sin(2 * Math.PI * 220 * t) + 0.5 * Math.sin(2 * Math.PI * 1300 * t)) +
      noise()
    );
  });
}

describe('audioLength', () => {
  it('drops trailing zeros only', () => {
    expect(audioLength(Float32Array.from([0, 1, 0, 2, 0, 0]))).toBe(4);
    expect(audioLength(new Float32Array(5))).toBe(0);
    expect(audioLength(new Float32Array(0))).toBe(0);
  });
});

describe('trimmedLength', () => {
  it('ends on the first frame that sees no audio', () => {
    // 5 s: frame f's window starts at 160f - 200, so frame 502 (at 80 120) is the first past 80 000.
    expect(trimmedLength(80_000, 480_000, GEOMETRY)).toBe(502 * 160 + 200);
  });

  it('gives up when the audio (nearly) fills the window', () => {
    expect(trimmedLength(480_000, 480_000, GEOMETRY)).toBeNull();
    expect(trimmedLength(479_500, 480_000, GEOMETRY)).toBeNull();
  });
});

describe('padFrames', () => {
  it('repeats each row’s last frame', () => {
    const padded = padFrames({ data: Float32Array.from([1, 2, 3, 4, 5, 6]), dims: [2, 3] }, 5);
    expect(padded.dims).toEqual([2, 5]);
    expect([...padded.data]).toEqual([1, 2, 3, 3, 3, 4, 5, 6, 6, 6]);
  });
});

describe('skipPaddingFrames', () => {
  const full = extractor();
  const patched = extractor();
  skipPaddingFrames(patched);

  const cases: [string, Float32Array][] = [
    ['an empty clip', new Float32Array(0)],
    ['one sample', Float32Array.from([0.25])],
    ['0.3 s', audio(0.3)],
    ['2 s', audio(2, 2)],
    ['5 s', audio(5, 3)],
    ['10 s', audio(10, 4)],
    ['an odd length', audio(7.31234, 5)],
    ['29.9 s', audio(29.9, 6)],
    ['30 s', audio(30, 7)],
    ['40 s (truncated to 30)', audio(40, 8)],
    ['digital silence', new Float32Array(16_000 * 3)],
    ['a near-silent clip (max - 8 below the -10 floor)', audio(4, 9).map((x) => x * 1e-6)],
    [
      'audio ending in exact zeros',
      Float32Array.from([...audio(3, 10), ...new Float32Array(8000)]),
    ],
  ];
  // ONNX Runtime's matmul (the mel projection) rounds a few values differently with fewer columns.
  // One float32 step near 1 is 1.2e-7; features span about -1.5 to 1.5.
  // Two full 30 s extractions per case: seconds each, more under a parallel test run.
  it.each(cases)(
    'matches the full extractor on %s',
    async (_, clip) => {
      const expected = (await full(clip)).input_features;
      const actual = (await patched(clip)).input_features;
      expect(actual.dims).toEqual(expected.dims);
      let largest = 0;
      expected.data.forEach((value, i) => {
        largest = Math.max(largest, Math.abs(value - (actual.data[i] ?? NaN)));
      });
      expect(largest).toBeLessThanOrEqual(5e-7);
    },
    30_000,
  );
});
