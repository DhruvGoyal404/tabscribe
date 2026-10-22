import type { Level } from './level';

/**
 * Every message sent with chrome.runtime.sendMessage. Receivers must check the sender and
 * run isMessage before acting; anything else is dropped.
 */
export type Message =
  /** popup -> service worker */
  | { type: 'capture/start'; tabId: number }
  | { type: 'capture/stop' }
  /** service worker -> offscreen document */
  | { type: 'offscreen/start'; streamId: string }
  | { type: 'offscreen/stop' }
  /** offscreen document -> service worker */
  | { type: 'offscreen/track-ended' }
  | { type: 'offscreen/error'; message: string };

/** Sent from the offscreen document to the popup over the LEVELS_PORT connection. */
export interface LevelMessage extends Level {
  type: 'level';
}

export type Reply = { ok: true } | { ok: false; error: string };

export const LEVELS_PORT = 'levels';

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

export function isMessage(value: unknown): value is Message {
  if (!isRecord(value)) return false;
  switch (value.type) {
    case 'capture/start':
      return (
        hasExactKeys(value, ['type', 'tabId']) &&
        Number.isSafeInteger(value.tabId) &&
        (value.tabId as number) >= 0
      );
    case 'capture/stop':
    case 'offscreen/stop':
    case 'offscreen/track-ended':
      return hasExactKeys(value, ['type']);
    case 'offscreen/start':
      return (
        hasExactKeys(value, ['type', 'streamId']) &&
        isBoundedString(value.streamId, MAX_STREAM_ID_LENGTH)
      );
    case 'offscreen/error':
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
