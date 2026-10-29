import { describe, expect, it } from 'vitest';
import { planBackend } from '../asr/backend';
import { describeError, isLevel, isLevelMessage, isMessage, isReply, settle } from './messages';

const plan = planBackend({
  gpu: { available: false, reason: 'none' },
  webgpuFailure: null,
  crossOriginIsolated: true,
  hardwareConcurrency: 8,
});
const request = { plan, allowDownload: false };

describe('isMessage', () => {
  it.each([
    { type: 'capture/start', tabId: 7, streamId: 'abc123', ...request },
    { type: 'capture/start', tabId: 0, streamId: 'abc123', ...request },
    { type: 'capture/stop' },
    { type: 'offscreen/start', streamId: 'abc123', muted: false, ...request },
    { type: 'offscreen/start', streamId: 'abc123', muted: true, ...request },
    { type: 'offscreen/set-muted', muted: true },
    { type: 'offscreen/set-muted', muted: false },
    { type: 'offscreen/stop' },
    { type: 'offscreen/track-ended' },
    { type: 'offscreen/error', message: 'NotAllowedError: denied' },
    { type: 'offscreen/webgpu-failed', message: 'Error: no adapter' },
    { type: 'capture/start', tabId: 1, streamId: 'abc123', plan, allowDownload: true },
  ])('accepts %o', (message) => {
    expect(isMessage(message)).toBe(true);
  });

  it.each([
    ['null', null],
    ['a string', 'capture/stop'],
    ['an array', [{ type: 'capture/stop' }]],
    ['an unknown type', { type: 'capture/pause' }],
    ['a missing type', { tabId: 1 }],
    ['a string tab id', { type: 'capture/start', tabId: '1', streamId: 'abc123', ...request }],
    ['a fractional tab id', { type: 'capture/start', tabId: 1.5, streamId: 'abc123', ...request }],
    ['a negative tab id', { type: 'capture/start', tabId: -1, streamId: 'abc123', ...request }],
    ['a NaN tab id', { type: 'capture/start', tabId: NaN, streamId: 'abc123', ...request }],
    ['a missing tab id', { type: 'capture/start', streamId: 'abc123', ...request }],
    ['a start without a stream id', { type: 'capture/start', tabId: 1, ...request }],
    [
      'a start with an empty stream id',
      { type: 'capture/start', tabId: 1, streamId: '', ...request },
    ],
    ['extra keys', { type: 'capture/stop', tabId: 1 }],
    ['an empty stream id', { type: 'offscreen/start', streamId: '', muted: false, ...request }],
    [
      'an oversized stream id',
      { type: 'offscreen/start', streamId: 'x'.repeat(513), muted: false, ...request },
    ],
    [
      'a start without a plan',
      { type: 'capture/start', tabId: 1, streamId: 'abc123', allowDownload: false },
    ],
    [
      'a start without the download flag',
      { type: 'capture/start', tabId: 1, streamId: 'abc123', plan },
    ],
    [
      'a start with a string download flag',
      { type: 'offscreen/start', streamId: 'abc123', muted: false, plan, allowDownload: 'yes' },
    ],
    [
      'a start with another model',
      {
        type: 'offscreen/start',
        streamId: 'abc123',
        muted: false,
        plan: { ...plan, dtype: { encoder: 'fp32', decoder: 'fp32' } },
        allowDownload: true,
      },
    ],
    ['an empty WebGPU failure', { type: 'offscreen/webgpu-failed', message: '' }],
    ['a start without the muted flag', { type: 'offscreen/start', streamId: 'abc123', ...request }],
    ['a non-boolean muted flag', { type: 'offscreen/set-muted', muted: 'yes' }],
    ['a set-muted without the flag', { type: 'offscreen/set-muted' }],
    ['a set-muted with extra keys', { type: 'offscreen/set-muted', muted: true, tabId: 1 }],
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
