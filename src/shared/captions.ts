// What the offscreen document tells the popup about captions, over the session port.
import { isCaptionPlan, type CaptionPlan } from '../asr/backend';

export type CaptionStatus =
  /** Loading the model: from cache, or downloading it (bytes so far of the total). */
  | {
      state: 'loading';
      plan: CaptionPlan;
      downloading: boolean;
      loadedBytes: number;
      totalBytes: number;
      /** Recent download speed; null until known. */
      bytesPerSecond: number | null;
    }
  /** The model is loaded and audio flows to the VAD. */
  | { state: 'listening'; plan: CaptionPlan; vad: 'silero' | 'energy' }
  /** Captions stopped; capture and playback go on. */
  | { state: 'error'; plan: CaptionPlan; message: string };

export interface Caption {
  /** A partial previews a segment still being spoken; a final replaces it. */
  kind: 'partial' | 'final';
  /** Of the (last) segment it covers. */
  segmentId: number;
  /** Empty for a final that was dropped (no speech, a loop, not English): clears the partial. */
  text: string;
}

export type CaptionMessage =
  | { type: 'captions/status'; status: CaptionStatus }
  | { type: 'captions/caption'; caption: Caption };

export const MAX_CAPTION_LENGTH = 10_000;
const MAX_STATUS_MESSAGE_LENGTH = 1000;

type UnknownRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const hasExactKeys = (value: UnknownRecord, keys: readonly string[]): boolean => {
  const actual = Object.keys(value);
  return actual.length === keys.length && actual.every((key) => keys.includes(key));
};

const isCount = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function isCaptionStatus(value: unknown): value is CaptionStatus {
  if (!isRecord(value) || !isCaptionPlan(value.plan)) return false;
  switch (value.state) {
    case 'loading':
      return (
        hasExactKeys(value, [
          'state',
          'plan',
          'downloading',
          'loadedBytes',
          'totalBytes',
          'bytesPerSecond',
        ]) &&
        typeof value.downloading === 'boolean' &&
        isCount(value.loadedBytes) &&
        isCount(value.totalBytes) &&
        (value.bytesPerSecond === null || isCount(value.bytesPerSecond))
      );
    case 'listening':
      return (
        hasExactKeys(value, ['state', 'plan', 'vad']) &&
        (value.vad === 'silero' || value.vad === 'energy')
      );
    case 'error':
      return (
        hasExactKeys(value, ['state', 'plan', 'message']) &&
        typeof value.message === 'string' &&
        value.message.length > 0 &&
        value.message.length <= MAX_STATUS_MESSAGE_LENGTH
      );
    default:
      return false;
  }
}

export const isCaption = (value: unknown): value is Caption =>
  isRecord(value) &&
  hasExactKeys(value, ['kind', 'segmentId', 'text']) &&
  (value.kind === 'partial' || value.kind === 'final') &&
  Number.isSafeInteger(value.segmentId) &&
  (value.segmentId as number) >= 0 &&
  typeof value.text === 'string' &&
  value.text.length <= MAX_CAPTION_LENGTH;

export function isCaptionMessage(value: unknown): value is CaptionMessage {
  if (!isRecord(value)) return false;
  switch (value.type) {
    case 'captions/status':
      return hasExactKeys(value, ['type', 'status']) && isCaptionStatus(value.status);
    case 'captions/caption':
      return hasExactKeys(value, ['type', 'caption']) && isCaption(value.caption);
    default:
      return false;
  }
}

/** A rough ETA: whole minutes, or 5 s steps under a minute and a half. */
function timeLeft(seconds: number): string {
  if (seconds < 90) return `about ${String(Math.max(5, Math.round(seconds / 5) * 5))} s left`;
  return `about ${String(Math.round(seconds / 60))} min left`;
}

/** One line for the popup. */
export function describeStatus(status: CaptionStatus): string {
  switch (status.state) {
    case 'loading': {
      if (!status.downloading || status.totalBytes === 0) return 'Loading the speech model…';
      const mb = (bytes: number) => (bytes / 1e6).toFixed(1);
      const progress = `Downloading the speech model: ${mb(status.loadedBytes)} of ${mb(status.totalBytes)} MB`;
      const rate = status.bytesPerSecond;
      if (!rate) return progress;
      const left = timeLeft(Math.max(0, status.totalBytes - status.loadedBytes) / rate);
      return `${progress}, ${mb(rate)} MB/s, ${left}`;
    }
    case 'listening':
      return 'Listening for speech';
    case 'error':
      return `Captions stopped: ${status.message}`;
  }
}
