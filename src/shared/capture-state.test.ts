import { describe, expect, it } from 'vitest';
import {
  capturedTabId,
  IDLE,
  isCaptureState,
  transition,
  type CaptureEvent,
  type CaptureState,
} from './capture-state';

const starting: CaptureState = { status: 'starting', tabId: 1 };
const capturing: CaptureState = { status: 'capturing', tabId: 1 };
const stopping: CaptureState = { status: 'stopping', tabId: 1 };
const failed: CaptureState = { status: 'error', message: 'boom' };

describe('transition', () => {
  it.each<[string, CaptureState, CaptureEvent, CaptureState]>([
    ['idle -> starting', IDLE, { type: 'start', tabId: 1 }, starting],
    ['error -> starting (retry)', failed, { type: 'start', tabId: 1 }, starting],
    ['starting -> capturing', starting, { type: 'started' }, capturing],
    ['starting -> stopping (cancel)', starting, { type: 'stop' }, stopping],
    ['capturing -> stopping', capturing, { type: 'stop' }, stopping],
    ['stopping -> idle', stopping, { type: 'stopped' }, IDLE],
    ['error -> idle (dismiss)', failed, { type: 'stop' }, IDLE],
    [
      'starting -> error',
      starting,
      { type: 'failed', message: 'x' },
      { status: 'error', message: 'x' },
    ],
    [
      'capturing -> error',
      capturing,
      { type: 'failed', message: 'x' },
      { status: 'error', message: 'x' },
    ],
    [
      'stopping -> error',
      stopping,
      { type: 'failed', message: 'x' },
      { status: 'error', message: 'x' },
    ],
  ])('%s', (_label, from, event, to) => {
    expect(transition(from, event)).toEqual(to);
  });

  it.each<[string, CaptureState, CaptureEvent]>([
    ['start while starting', starting, { type: 'start', tabId: 2 }],
    ['start while capturing', capturing, { type: 'start', tabId: 2 }],
    ['start while stopping', stopping, { type: 'start', tabId: 2 }],
    ['started while idle', IDLE, { type: 'started' }],
    ['started while capturing', capturing, { type: 'started' }],
    ['stop while idle', IDLE, { type: 'stop' }],
    ['stop while stopping', stopping, { type: 'stop' }],
    ['stopped while capturing', capturing, { type: 'stopped' }],
    ['stopped while idle', IDLE, { type: 'stopped' }],
    ['failed while idle', IDLE, { type: 'failed', message: 'x' }],
  ])('ignores %s', (_label, from, event) => {
    expect(transition(from, event)).toBe(from);
  });

  it('does not mutate its input', () => {
    const before = { ...capturing };
    transition(capturing, { type: 'stop' });
    expect(capturing).toEqual(before);
  });
});

describe('capturedTabId', () => {
  it('returns the tab for active states only', () => {
    expect(capturedTabId(capturing)).toBe(1);
    expect(capturedTabId(stopping)).toBe(1);
    expect(capturedTabId(IDLE)).toBeUndefined();
    expect(capturedTabId(failed)).toBeUndefined();
  });
});

describe('isCaptureState', () => {
  it.each([IDLE, starting, capturing, stopping, failed])('accepts %o', (value) => {
    expect(isCaptureState(value)).toBe(true);
  });

  it.each([
    undefined,
    null,
    {},
    { status: 'paused' },
    { status: 'capturing' },
    { status: 'capturing', tabId: '1' },
    { status: 'idle', tabId: 1 },
    { status: 'error' },
  ])('rejects %o', (value) => {
    expect(isCaptureState(value)).toBe(false);
  });
});
