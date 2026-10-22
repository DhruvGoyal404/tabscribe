/** Capture lifecycle, owned by the service worker and mirrored in chrome.storage.session. */
export type CaptureState =
  | { status: 'idle' }
  | { status: 'starting'; tabId: number }
  | { status: 'capturing'; tabId: number }
  | { status: 'stopping'; tabId: number }
  | { status: 'error'; message: string };

export type CaptureEvent =
  | { type: 'start'; tabId: number }
  | { type: 'started' }
  | { type: 'stop' }
  | { type: 'stopped' }
  | { type: 'failed'; message: string };

export const STATE_KEY = 'captureState';
export const IDLE: CaptureState = { status: 'idle' };

/** Pure reducer. Events that don't apply to the current state leave it unchanged. */
export function transition(state: CaptureState, event: CaptureEvent): CaptureState {
  switch (event.type) {
    case 'start':
      return state.status === 'idle' || state.status === 'error'
        ? { status: 'starting', tabId: event.tabId }
        : state;
    case 'started':
      return state.status === 'starting' ? { status: 'capturing', tabId: state.tabId } : state;
    case 'stop':
      if (state.status === 'starting' || state.status === 'capturing') {
        return { status: 'stopping', tabId: state.tabId };
      }
      // Stopping from an error dismisses it.
      return state.status === 'error' ? IDLE : state;
    case 'stopped':
      return state.status === 'stopping' ? IDLE : state;
    case 'failed':
      return state.status === 'idle' ? state : { status: 'error', message: event.message };
  }
}

export function capturedTabId(state: CaptureState): number | undefined {
  return 'tabId' in state ? state.tabId : undefined;
}

export function isCaptureState(value: unknown): value is CaptureState {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort().join(',');
  switch (record.status) {
    case 'idle':
      return keys === 'status';
    case 'starting':
    case 'capturing':
    case 'stopping':
      return keys === 'status,tabId' && Number.isSafeInteger(record.tabId);
    case 'error':
      return keys === 'message,status' && typeof record.message === 'string';
    default:
      return false;
  }
}
