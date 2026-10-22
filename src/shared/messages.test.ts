import { describe, expect, it } from 'vitest';
import { describeError, isLevel, isLevelMessage, isMessage, isReply, settle } from './messages';

describe('isMessage', () => {
  it.each([
    { type: 'capture/start', tabId: 7 },
    { type: 'capture/start', tabId: 0 },
    { type: 'capture/stop' },
    { type: 'offscreen/start', streamId: 'abc123' },
    { type: 'offscreen/stop' },
    { type: 'offscreen/track-ended' },
    { type: 'offscreen/error', message: 'NotAllowedError: denied' },
  ])('accepts %o', (message) => {
    expect(isMessage(message)).toBe(true);
  });

  it.each([
    ['null', null],
    ['a string', 'capture/stop'],
    ['an array', [{ type: 'capture/stop' }]],
    ['an unknown type', { type: 'capture/pause' }],
    ['a missing type', { tabId: 1 }],
    ['a string tab id', { type: 'capture/start', tabId: '1' }],
    ['a fractional tab id', { type: 'capture/start', tabId: 1.5 }],
    ['a negative tab id', { type: 'capture/start', tabId: -1 }],
    ['a NaN tab id', { type: 'capture/start', tabId: NaN }],
    ['a missing tab id', { type: 'capture/start' }],
    ['extra keys', { type: 'capture/stop', tabId: 1 }],
    ['an empty stream id', { type: 'offscreen/start', streamId: '' }],
    ['an oversized stream id', { type: 'offscreen/start', streamId: 'x'.repeat(513) }],
    ['a non-string error', { type: 'offscreen/error', message: 42 }],
  ])('rejects %s', (_label, message) => {
    expect(isMessage(message)).toBe(false);
  });
});

describe('isLevel / isLevelMessage', () => {
  const level = { rmsDbfs: -20, peakDbfs: -6, clipped: false };

  it('accepts well-formed levels', () => {
    expect(isLevel(level)).toBe(true);
    expect(isLevelMessage({ type: 'level', ...level })).toBe(true);
  });

  it.each([
    ['non-finite RMS', { ...level, rmsDbfs: -Infinity }],
    ['NaN peak', { ...level, peakDbfs: NaN }],
    ['string clipped flag', { ...level, clipped: 'no' }],
    ['extra keys', { ...level, extra: 1 }],
    ['missing keys', { rmsDbfs: -20 }],
  ])('rejects %s', (_label, value) => {
    expect(isLevel(value)).toBe(false);
    expect(isLevelMessage({ type: 'level', ...value })).toBe(false);
  });

  it('requires the level type tag', () => {
    expect(isLevelMessage(level)).toBe(false);
    expect(isLevelMessage({ type: 'other', ...level })).toBe(false);
  });
});

describe('isReply', () => {
  it('accepts ok and error replies', () => {
    expect(isReply({ ok: true })).toBe(true);
    expect(isReply({ ok: false, error: 'boom' })).toBe(true);
  });

  it.each([undefined, {}, { ok: 'yes' }, { ok: false }, { ok: true, error: 'x' }])(
    'rejects %o',
    (value) => {
      expect(isReply(value)).toBe(false);
    },
  );
});

describe('settle / describeError', () => {
  it('turns success into an ok reply', async () => {
    await expect(settle(Promise.resolve(1))).resolves.toEqual({ ok: true });
  });

  it('turns failure into an error reply that passes isReply', async () => {
    const reply = await settle(Promise.reject(new TypeError('bad stream')));
    expect(reply).toEqual({ ok: false, error: 'TypeError: bad stream' });
    expect(isReply(reply)).toBe(true);
  });

  it('bounds and defaults error text', () => {
    expect(describeError('x'.repeat(5000))).toHaveLength(1000);
    expect(describeError('')).toBe('Unknown error');
  });
});
