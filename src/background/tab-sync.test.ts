import { describe, expect, it } from 'vitest';
import { IDLE, type CaptureState } from '../shared/capture-state';
import { mutedChange } from './tab-sync';

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
