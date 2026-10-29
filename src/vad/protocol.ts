// Messages for the VAD worker. Three channels, so audio never passes through a page's main thread
// and VAD never waits on Whisper:
// - control: the spawning page <-> the worker (worker.postMessage);
// - audio:   the capture worklet -> the worker (a MessagePort handed over in `init`);
// - segments: the worker -> the ASR worker (another MessagePort handed over in `init`).
// Messages on different channels aren't ordered relative to each other, so end-of-stream travels
// on the data path: `flush` follows the last audio chunk, `flushed` follows the last segment.
import type { EndReason, SegmentEvent, SegmenterOptions } from './segmenter';

export type VadKind = 'silero' | 'energy';

/** Segmenter settings a caller may tune; the rest are fixed. Values are milliseconds. */
export const TUNABLE = [
  'minSilenceMs',
  'maxSegmentMs',
  'cutSearchMs',
  'overlapMs',
  'firstPartialMs',
  'partialIntervalMs',
] as const satisfies readonly (keyof SegmenterOptions)[];
export type SegmenterTuning = Partial<Pick<SegmenterOptions, (typeof TUNABLE)[number]>>;

/** Bounds for tunable values: 50 ms to 30 s (Whisper's window). */
const MIN_TUNING_MS = 50;
const MAX_TUNING_MS = 30_000;

export type VadRequest =
  | {
      type: 'init';
      id: number;
      /** Sample rate of the audio port's chunks (the capture AudioContext's rate). */
      inputRate: number;
      /** 'silero' falls back to 'energy' if the model fails to load. */
      vad: VadKind;
      tuning: SegmenterTuning;
      audio: MessagePort;
      segments: MessagePort;
    }
  | { type: 'stats'; id: number };

export interface VadStats {
  frames: number;
  speechFrames: number;
  /** Mean VAD inference time per 32 ms frame. */
  msPerFrame: number;
  segments: number;
  partials: number;
}

export type VadResponse =
  | { type: 'ready'; id: number; vad: VadKind; fallbackReason: string | null }
  | { type: 'stats'; id: number; stats: VadStats }
  | { type: 'error'; id: number; message: string };

/** On the audio port: mono chunks at `inputRate`, then `flush` when capture stops. */
export type AudioPortMessage = { type: 'audio'; samples: Float32Array } | { type: 'flush' };

/** On the segments port: segments in order, then `flushed` once the stream has been drained. */
export type SegmentPortMessage = SegmentEvent | { type: 'flushed'; stats: VadStats };

/** Longest audio chunk accepted: one second at 192 kHz. */
const MAX_CHUNK_SAMPLES = 192_000;

type UnknownRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const hasExactKeys = (value: UnknownRecord, keys: readonly string[]): boolean => {
  const actual = Object.keys(value);
  return actual.length === keys.length && actual.every((key) => keys.includes(key));
};

const isId = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) >= 0;
const isCount = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;
const isSampleIndex = (value: unknown): value is number =>
  Number.isSafeInteger(value) && Number(value) >= 0;

function isTuning(value: unknown): value is SegmenterTuning {
  if (!isRecord(value)) return false;
  return Object.entries(value).every(
    ([key, ms]) =>
      (TUNABLE as readonly string[]).includes(key) &&
      typeof ms === 'number' &&
      ms >= MIN_TUNING_MS &&
      ms <= MAX_TUNING_MS,
  );
}

export function isVadRequest(value: unknown): value is VadRequest {
  if (!isRecord(value) || !isId(value.id)) return false;
  switch (value.type) {
    case 'init':
      return (
        hasExactKeys(value, ['type', 'id', 'inputRate', 'vad', 'tuning', 'audio', 'segments']) &&
        Number.isSafeInteger(value.inputRate) &&
        Number(value.inputRate) >= 8_000 &&
        Number(value.inputRate) <= 192_000 &&
        (value.vad === 'silero' || value.vad === 'energy') &&
        isTuning(value.tuning) &&
        value.audio instanceof MessagePort &&
        value.segments instanceof MessagePort
      );
    case 'stats':
      return hasExactKeys(value, ['type', 'id']);
    default:
      return false;
  }
}

const isStats = (value: unknown): value is VadStats =>
  isRecord(value) &&
  hasExactKeys(value, ['frames', 'speechFrames', 'msPerFrame', 'segments', 'partials']) &&
  isSampleIndex(value.frames) &&
  isSampleIndex(value.speechFrames) &&
  isCount(value.msPerFrame) &&
  isSampleIndex(value.segments) &&
  isSampleIndex(value.partials);

export function isVadResponse(value: unknown): value is VadResponse {
  if (!isRecord(value) || !isId(value.id)) return false;
  switch (value.type) {
    case 'ready':
      return (
        hasExactKeys(value, ['type', 'id', 'vad', 'fallbackReason']) &&
        (value.vad === 'silero' || value.vad === 'energy') &&
        (value.fallbackReason === null ||
          (typeof value.fallbackReason === 'string' && value.fallbackReason.length <= 1000))
      );
    case 'stats':
      return hasExactKeys(value, ['type', 'id', 'stats']) && isStats(value.stats);
    case 'error':
      return (
        hasExactKeys(value, ['type', 'id', 'message']) &&
        typeof value.message === 'string' &&
        value.message.length <= 1000
      );
    default:
      return false;
  }
}

export function isAudioPortMessage(value: unknown): value is AudioPortMessage {
  if (!isRecord(value)) return false;
  if (value.type === 'flush') return hasExactKeys(value, ['type']);
  return (
    hasExactKeys(value, ['type', 'samples']) &&
    value.type === 'audio' &&
    value.samples instanceof Float32Array &&
    value.samples.length > 0 &&
    value.samples.length <= MAX_CHUNK_SAMPLES
  );
}

const END_REASONS: readonly EndReason[] = ['silence', 'max-length', 'flush'];

/** Segments are at most one Whisper window long. */
const MAX_SEGMENT_SAMPLES = 16_000 * 30;

export function isSegmentEvent(value: unknown): value is SegmentEvent {
  if (!isRecord(value)) return false;
  const { segmentId, startSample, endSample, audio } = value;
  if (
    !isSampleIndex(segmentId) ||
    !isSampleIndex(startSample) ||
    !isSampleIndex(endSample) ||
    !(audio instanceof Float32Array) ||
    audio.length === 0 ||
    audio.length > MAX_SEGMENT_SAMPLES ||
    audio.length !== endSample - startSample
  ) {
    return false;
  }
  const { overlapSamples } = value;
  if (!isSampleIndex(overlapSamples) || overlapSamples > audio.length) return false;
  switch (value.type) {
    case 'partial':
      return hasExactKeys(value, [
        'type',
        'segmentId',
        'startSample',
        'endSample',
        'overlapSamples',
        'audio',
      ]);
    case 'final': {
      const { speechEndSample, reason } = value;
      return (
        hasExactKeys(value, [
          'type',
          'segmentId',
          'startSample',
          'endSample',
          'speechEndSample',
          'reason',
          'overlapSamples',
          'audio',
        ]) &&
        isSampleIndex(speechEndSample) &&
        (END_REASONS as readonly unknown[]).includes(reason)
      );
    }
    default:
      return false;
  }
}

export function isSegmentPortMessage(value: unknown): value is SegmentPortMessage {
  if (isRecord(value) && value.type === 'flushed') {
    return hasExactKeys(value, ['type', 'stats']) && isStats(value.stats);
  }
  return isSegmentEvent(value);
}
