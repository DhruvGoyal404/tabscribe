// Per-call cost of one Whisper window, split into encoder and decoder time. P3b sizes its segments
// from this: the encoder always sees a padded 30 s window, while decoding scales with the tokens
// produced, so short segments are only cheap if the encoder is.
import { ASR_SAMPLE_RATE } from '../asr/resample';

export const PROFILE_SECONDS = [2, 5, 10, 15] as const;
/** Different stretches of the clip per length, so one unusual sentence doesn't decide the result. */
export const PROFILE_OFFSETS = 3;

export interface CallTiming {
  featuresMs: number;
  encoderMs: number;
  decoderMs: number;
  decoderSteps: number;
  /** What was decoded, to check by eye that the timed decode was real. */
  text: string;
}

export interface CostRow {
  config: string;
  seconds: number;
  /** Medians over the measured calls. */
  featuresMs: number;
  encoderMs: number;
  decoderMs: number;
  decoderSteps: number;
  /** Decoder time per step, from the summed time and steps of all calls. */
  msPerStep: number;
  totalMs: number;
  /** Total call time / segment duration. */
  callRtf: number;
  calls: number;
  texts: string[];
}

export function median(values: readonly number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? (sorted[mid] ?? NaN)
    : ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
}

/** The nearest-rank percentile (`p` from 0 to 100): a value that occurred, not an interpolation. */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.max(1, Math.ceil((p / 100) * sorted.length));
  return sorted[rank - 1] ?? NaN;
}

/** Start samples of `count` segments spread evenly from the start to the end of the clip. */
export function segmentOffsets(
  totalSamples: number,
  segmentSamples: number,
  count: number,
): number[] {
  const span = totalSamples - segmentSamples;
  if (span < 0) return [];
  if (count <= 1 || span === 0) return [0];
  return Array.from({ length: count }, (_, i) => Math.round((span * i) / (count - 1)));
}

export function summarise(
  config: string,
  seconds: number,
  timings: readonly CallTiming[],
): CostRow {
  const totals = timings.map((t) => t.featuresMs + t.encoderMs + t.decoderMs);
  const steps = timings.reduce((sum, t) => sum + t.decoderSteps, 0);
  const decoderTotal = timings.reduce((sum, t) => sum + t.decoderMs, 0);
  const totalMs = median(totals);
  return {
    config,
    seconds,
    featuresMs: median(timings.map((t) => t.featuresMs)),
    encoderMs: median(timings.map((t) => t.encoderMs)),
    decoderMs: median(timings.map((t) => t.decoderMs)),
    decoderSteps: median(timings.map((t) => t.decoderSteps)),
    msPerStep: steps > 0 ? decoderTotal / steps : NaN,
    totalMs,
    callRtf: totalMs / 1000 / seconds,
    calls: timings.length,
    texts: timings.map((t) => t.text),
  };
}

export const secondsToSamples = (seconds: number): number => Math.round(seconds * ASR_SAMPLE_RATE);
