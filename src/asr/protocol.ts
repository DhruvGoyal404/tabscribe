// Messages between a page (the offscreen document in P3b, the bench page in P3a) and the ASR
// worker it spawns. Only the spawning page can post to a dedicated worker, but both sides still
// validate every message so a bug can't turn into an unexpected download or a confusing crash.
import {
  isDevice,
  isDtype,
  isModelKey,
  type Device,
  type DtypeSpec,
  type ModelKey,
} from './models';

export type AsrTask = 'transcribe' | 'translate';

/** Longest clip accepted in one request: 10 minutes of 16 kHz audio. */
export const MAX_AUDIO_SAMPLES = 16_000 * 600;
/** Whisper's window: one encoder pass covers at most 30 s of 16 kHz audio. */
export const MAX_WINDOW_SAMPLES = 16_000 * 30;

export type WorkerRequest =
  | { type: 'gpu-info'; id: number }
  /** Downloads (or reads from cache) and initialises a pinned model. */
  | { type: 'load'; id: number; model: ModelKey; device: Device; dtype: DtypeSpec }
  /** 16 kHz mono. `language: null` detects the language from the first 30 s. */
  | { type: 'transcribe'; id: number; audio: Float32Array; task: AsrTask; language: string | null }
  /**
   * Times one English decode of a single window (at most 30 s) in its parts: feature extraction,
   * the encoder session, then greedy decoding from the encoder output.
   */
  | { type: 'profile'; id: number; audio: Float32Array }
  | { type: 'unload'; id: number };

export type GpuInfo =
  | {
      available: true;
      vendor: string;
      architecture: string;
      device: string;
      description: string;
      isFallbackAdapter: boolean;
      shaderF16: boolean;
      maxBufferSize: number;
      maxStorageBufferBindingSize: number;
    }
  | { available: false; reason: string };

export type WorkerResponse =
  | { type: 'gpu-info'; id: number; info: GpuInfo }
  | { type: 'progress'; id: number; file: string; loaded: number; total: number }
  | { type: 'loaded'; id: number; ms: number }
  | {
      type: 'result';
      id: number;
      text: string;
      /** The language decoded with: forced, or detected. */
      language: string;
      /** Time spent on language detection; null when the language was forced. */
      detectMs: number | null;
      /** Total processing time, including detection. */
      ms: number;
    }
  | {
      type: 'profile';
      id: number;
      featuresMs: number;
      encoderMs: number;
      decoderMs: number;
      /** Decoder steps run, including the end-of-text token (excludes the 4 prompt tokens). */
      decoderSteps: number;
      text: string;
    }
  | { type: 'unloaded'; id: number }
  | { type: 'error'; id: number; message: string };

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
const isShortString = (value: unknown, max = 1000): value is string =>
  typeof value === 'string' && value.length <= max;

/** Whisper language codes are 2-3 lower-case letters ("en", "hi", "haw"). */
export const isLanguageCode = (value: unknown): value is string =>
  typeof value === 'string' && /^[a-z]{2,3}$/.test(value);

const isDtypeSpec = (value: unknown): value is DtypeSpec =>
  isRecord(value) &&
  hasExactKeys(value, ['encoder', 'decoder']) &&
  isDtype(value.encoder) &&
  isDtype(value.decoder);

export function isWorkerRequest(value: unknown): value is WorkerRequest {
  if (!isRecord(value) || !isId(value.id)) return false;
  switch (value.type) {
    case 'gpu-info':
    case 'unload':
      return hasExactKeys(value, ['type', 'id']);
    case 'load':
      return (
        hasExactKeys(value, ['type', 'id', 'model', 'device', 'dtype']) &&
        isModelKey(value.model) &&
        isDevice(value.device) &&
        isDtypeSpec(value.dtype)
      );
    case 'transcribe':
      return (
        hasExactKeys(value, ['type', 'id', 'audio', 'task', 'language']) &&
        value.audio instanceof Float32Array &&
        value.audio.length > 0 &&
        value.audio.length <= MAX_AUDIO_SAMPLES &&
        (value.task === 'transcribe' || value.task === 'translate') &&
        (value.language === null || isLanguageCode(value.language))
      );
    case 'profile':
      return (
        hasExactKeys(value, ['type', 'id', 'audio']) &&
        value.audio instanceof Float32Array &&
        value.audio.length > 0 &&
        value.audio.length <= MAX_WINDOW_SAMPLES
      );
    default:
      return false;
  }
}

const GPU_INFO_KEYS = [
  'available',
  'vendor',
  'architecture',
  'device',
  'description',
  'isFallbackAdapter',
  'shaderF16',
  'maxBufferSize',
  'maxStorageBufferBindingSize',
];

const isGpuInfo = (value: unknown): value is GpuInfo => {
  if (!isRecord(value)) return false;
  if (value.available === false) {
    return hasExactKeys(value, ['available', 'reason']) && isShortString(value.reason);
  }
  return (
    value.available === true &&
    hasExactKeys(value, GPU_INFO_KEYS) &&
    isShortString(value.vendor) &&
    isShortString(value.architecture) &&
    isShortString(value.device) &&
    isShortString(value.description) &&
    typeof value.isFallbackAdapter === 'boolean' &&
    typeof value.shaderF16 === 'boolean' &&
    isCount(value.maxBufferSize) &&
    isCount(value.maxStorageBufferBindingSize)
  );
};

export function isWorkerResponse(value: unknown): value is WorkerResponse {
  if (!isRecord(value) || !isId(value.id)) return false;
  switch (value.type) {
    case 'gpu-info':
      return hasExactKeys(value, ['type', 'id', 'info']) && isGpuInfo(value.info);
    case 'progress':
      return (
        hasExactKeys(value, ['type', 'id', 'file', 'loaded', 'total']) &&
        isShortString(value.file) &&
        isCount(value.loaded) &&
        isCount(value.total)
      );
    case 'loaded':
      return hasExactKeys(value, ['type', 'id', 'ms']) && isCount(value.ms);
    case 'result':
      return (
        hasExactKeys(value, ['type', 'id', 'text', 'language', 'detectMs', 'ms']) &&
        isShortString(value.text, 100_000) &&
        isLanguageCode(value.language) &&
        (value.detectMs === null || isCount(value.detectMs)) &&
        isCount(value.ms)
      );
    case 'profile':
      return (
        hasExactKeys(value, [
          'type',
          'id',
          'featuresMs',
          'encoderMs',
          'decoderMs',
          'decoderSteps',
          'text',
        ]) &&
        isCount(value.featuresMs) &&
        isCount(value.encoderMs) &&
        isCount(value.decoderMs) &&
        Number.isSafeInteger(value.decoderSteps) &&
        isCount(value.decoderSteps) &&
        isShortString(value.text, 100_000)
      );
    case 'unloaded':
      return hasExactKeys(value, ['type', 'id']);
    case 'error':
      return hasExactKeys(value, ['type', 'id', 'message']) && isShortString(value.message);
    default:
      return false;
  }
}
