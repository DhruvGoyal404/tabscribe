import { describe, expect, it } from 'vitest';
import { IDLE, type CaptureState } from '../shared/capture-state';
import { badgeRefresh, badgeTabId, badgeUpdates, mutedChange } from './tab-sync';

const capturing: CaptureState = { status: 'capturing', tabId: 7 };

describe('mutedChange', () => {
  it('returns the new muted state for the captured tab', () => {
    expect(mutedChange(capturing, 7, { mutedInfo: { muted: true } })).toBe(true);
    expect(mutedChange(capturing, 7, { mutedInfo: { muted: false } })).toBe(false);
  });

  it('ignores other tabs', () => {
    expect(mutedChange(capturing, 8, { mutedInfo: { muted: true } })).toBeUndefined();
  });

  it('ignores changes that are not about mute', () => {
    expect(mutedChange(capturing, 7, {})).toBeUndefined();
  });

  it.each<[string, CaptureState]>([
    ['idle', IDLE],
    ['starting', { status: 'starting', tabId: 7 }],
    ['stopping', { status: 'stopping', tabId: 7 }],
    ['error', { status: 'error', message: 'boom' }],
  ])('ignores mute changes while %s', (_label, state) => {
    expect(mutedChange(state, 7, { mutedInfo: { muted: true } })).toBeUndefined();
  });
});

describe('badgeTabId', () => {
  it('is the captured tab only while capturing', () => {
    expect(badgeTabId(capturing)).toBe(7);
    expect(badgeTabId({ status: 'starting', tabId: 7 })).toBeUndefined();
    expect(badgeTabId({ status: 'stopping', tabId: 7 })).toBeUndefined();
    expect(badgeTabId(IDLE)).toBeUndefined();
  });
});

describe('badgeUpdates', () => {
  it.each<[string, CaptureState, CaptureState, { tabId: number; text: string }[]]>([
    [
      'starting -> capturing sets the tab',
      { status: 'starting', tabId: 7 },
      capturing,
      [{ tabId: 7, text: 'ON' }],
    ],
    [
      'user stop clears the tab',
      capturing,
      { status: 'stopping', tabId: 7 },
      [{ tabId: 7, text: '' }],
    ],
    [
      'failure clears the tab',
      capturing,
      { status: 'error', message: 'boom' },
      [{ tabId: 7, text: '' }],
    ],
    ['idle -> starting changes nothing', IDLE, { status: 'starting', tabId: 7 }, []],
    ['stopping -> idle changes nothing', { status: 'stopping', tabId: 7 }, IDLE, []],
    ['a no-op transition changes nothing', capturing, capturing, []],
    [
      'switching tabs clears the old tab before setting the new one',
      capturing,
      { status: 'capturing', tabId: 8 },
      [
        { tabId: 7, text: '' },
        { tabId: 8, text: 'ON' },
      ],
    ],
  ])('%s', (_label, previous, next, expected) => {
    expect(badgeUpdates(previous, next)).toEqual(expected);
  });

  it('never leaves a badge behind across a full tab switch', () => {
    // Capture tab 7, then "Capture this tab" on 8: stop 7, start 8.
    const states: CaptureState[] = [
      IDLE,
      { status: 'starting', tabId: 7 },
      capturing,
      { status: 'stopping', tabId: 7 },
      IDLE,
      { status: 'starting', tabId: 8 },
      { status: 'capturing', tabId: 8 },
      { status: 'stopping', tabId: 8 },
      IDLE,
    ];
    const shown = new Map<number, string>();
    for (let i = 1; i < states.length; i++) {
      const previous = states[i - 1];
      const next = states[i];
      if (!previous || !next) throw new Error('unreachable');
      for (const { tabId, text } of badgeUpdates(previous, next)) shown.set(tabId, text);
      const badged = [...shown].filter(([, text]) => text !== '').map(([tabId]) => tabId);
      expect(badged).toEqual(next.status === 'capturing' ? [next.tabId] : []);
    }
  });
});

describe('badgeRefresh', () => {
  it.each(['loading', 'complete'])('re-applies the badge on %s for the captured tab', (status) => {
    expect(badgeRefresh(capturing, 7, { status })).toEqual({ tabId: 7, text: 'ON' });
  });

  it('ignores other tabs', () => {
    expect(badgeRefresh(capturing, 8, { status: 'loading' })).toBeUndefined();
  });

  it('ignores changes without a status', () => {
    expect(badgeRefresh(capturing, 7, { mutedInfo: { muted: true } })).toBeUndefined();
  });

  it.each<[string, CaptureState]>([
    ['idle', IDLE],
    ['starting', { status: 'starting', tabId: 7 }],
    ['stopping', { status: 'stopping', tabId: 7 }],
    ['error', { status: 'error', message: 'boom' }],
  ])('does not badge a tab while %s', (_label, state) => {
    expect(badgeRefresh(state, 7, { status: 'complete' })).toBeUndefined();
  });
});
