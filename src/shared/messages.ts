import { isCaptionPlan, type CaptionPlan } from '../asr/backend';
import type { Level } from './level';

/**
 * Every message sent with chrome.runtime.sendMessage. Receivers must check the sender and
 * run isMessage before acting; anything else is dropped.
 */
export type Message =
  /**
   * popup -> service worker. The popup gets the stream id: an id issued by the service worker
   * (which isn't cross-origin isolated) fails in the isolated offscreen document. It also plans
   * the caption backend, and `allowDownload` is true only when its Start button showed the
   * download size (the user's agreement to that download).
   */
  | ({ type: 'capture/start'; tabId: number; streamId: string } & CaptionRequest)
  | { type: 'capture/stop' }
  /** service worker -> offscreen document */
  | ({ type: 'offscreen/start'; streamId: string; muted: boolean } & CaptionRequest)
  | { type: 'offscreen/set-muted'; muted: boolean }
  | { type: 'offscreen/stop' }
  /** offscreen document -> service worker */
  | { type: 'offscreen/track-ended' }
  | { type: 'offscreen/error'; message: string }
  /** The planned WebGPU backend failed to load: plan WASM from now on (kept in storage.local). */
  | { type: 'offscreen/webgpu-failed'; message: string };

export interface CaptionRequest {
  plan: CaptionPlan;
  allowDownload: boolean;
}

/** Where the popup keeps a WebGPU load failure, so the next plan uses WASM. */
export const WEBGPU_FAILURE_KEY = 'webgpuFailure';

/** Sent from the offscreen document to the popup over the SESSION_PORT connection. */
export interface LevelMessage extends Level {
  type: 'level';
}

export type Reply = { ok: true } | { ok: false; error: string };

/** Offscreen document -> popup: levels, and caption status and captions (captions.ts). */
export const SESSION_PORT = 'session';

const MAX_STREAM_ID_LENGTH = 512;
const MAX_ERROR_LENGTH = 1000;

type UnknownRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const hasExactKeys = (value: UnknownRecord, keys: readonly string[]): boolean => {
  const actual = Object.keys(value);
  return actual.length === keys.length && actual.every((key) => keys.includes(key));
};

const isBoundedString = (value: unknown, max: number): value is string =>
  typeof value === 'string' && value.length > 0 && value.length <= max;

const hasCaptionRequest = (value: UnknownRecord): boolean =>
  isCaptionPlan(value.plan) && typeof value.allowDownload === 'boolean';

export function isMessage(value: unknown): value is Message {
  if (!isRecord(value)) return false;
  switch (value.type) {
    case 'capture/start':
      return (
        hasExactKeys(value, ['type', 'tabId', 'streamId', 'plan', 'allowDownload']) &&
        Number.isSafeInteger(value.tabId) &&
        (value.tabId as number) >= 0 &&
        isBoundedString(value.streamId, MAX_STREAM_ID_LENGTH) &&
        hasCaptionRequest(value)
      );
    case 'capture/stop':
    case 'offscreen/stop':
    case 'offscreen/track-ended':
      return hasExactKeys(value, ['type']);
    case 'offscreen/start':
      return (
        hasExactKeys(value, ['type', 'streamId', 'muted', 'plan', 'allowDownload']) &&
        isBoundedString(value.streamId, MAX_STREAM_ID_LENGTH) &&
        typeof value.muted === 'boolean' &&
        hasCaptionRequest(value)
      );
    case 'offscreen/set-muted':
      return hasExactKeys(value, ['type', 'muted']) && typeof value.muted === 'boolean';
    case 'offscreen/error':
    case 'offscreen/webgpu-failed':
      return (
        hasExactKeys(value, ['type', 'message']) && isBoundedString(value.message, MAX_ERROR_LENGTH)
      );
    default:
      return false;
  }
}

export function isLevel(value: unknown): value is Level {
  return (
    isRecord(value) &&
    hasExactKeys(value, ['rmsDbfs', 'peakDbfs', 'clipped']) &&
    Number.isFinite(value.rmsDbfs) &&
    Number.isFinite(value.peakDbfs) &&
    typeof value.clipped === 'boolean'
  );
}

export function isLevelMessage(value: unknown): value is LevelMessage {
  if (!isRecord(value) || value.type !== 'level') return false;
  const level = { ...value };
  delete level.type;
  return isLevel(level);
}

export function isReply(value: unknown): value is Reply {
  if (!isRecord(value)) return false;
  if (value.ok === true) return hasExactKeys(value, ['ok']);
  return (
    value.ok === false &&
    hasExactKeys(value, ['ok', 'error']) &&
    isBoundedString(value.error, MAX_ERROR_LENGTH)
  );
}

export function describeError(error: unknown): string {
  const text = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  return text.slice(0, MAX_ERROR_LENGTH) || 'Unknown error';
}

/** Resolves to a Reply instead of rejecting, for answering via sendResponse. */
export async function settle(task: Promise<unknown>): Promise<Reply> {
  try {
    await task;
    return { ok: true };
  } catch (error) {
    return { ok: false, error: describeError(error) };
  }
}
