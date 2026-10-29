import { describe, expect, it } from 'vitest';
import {
  isAudioPortMessage,
  isSegmentEvent,
  isSegmentPortMessage,
  isVadRequest,
  isVadResponse,
} from './protocol';

const { port1, port2 } = new MessageChannel();
const init = {
  type: 'init',
  id: 1,
  inputRate: 48_000,
  vad: 'silero',
  tuning: {},
  audio: port1,
  segments: port2,
};

describe('isVadRequest', () => {
  it.each([
    init,
    { ...init, vad: 'energy', inputRate: 44_100 },
    { ...init, tuning: { maxSegmentMs: 10_000, minSilenceMs: 500, partialIntervalMs: 2_000 } },
    { ...init, tuning: { cutSearchMs: 0, overlapMs: 0 } },
    { type: 'stats', id: 3 },
  ])('accepts %o', (request) => {
    expect(isVadRequest(request)).toBe(true);
  });

  it.each([
    ['a missing port', { ...init, segments: undefined }],
    ['a port that is not a MessagePort', { ...init, audio: {} }],
    ['an unknown VAD', { ...init, vad: 'webrtc' }],
    ['a silly sample rate', { ...init, inputRate: 1_000 }],
    ['a fractional sample rate', { ...init, inputRate: 44_100.5 }],
    ['an untunable setting', { ...init, tuning: { startThreshold: 0.1 } }],
    ['an out-of-range setting', { ...init, tuning: { maxSegmentMs: 60_000 } }],
    ['a zero that only some settings allow', { ...init, tuning: { maxSegmentMs: 0 } }],
    ['a cut search under 50 ms but not 0', { ...init, tuning: { cutSearchMs: 10 } }],
    ['a string setting', { ...init, tuning: { overlapMs: '2000' } }],
    ['extra keys', { type: 'stats', id: 2, now: true }],
    ['flush on the control channel (it belongs on the audio port)', { type: 'flush', id: 2 }],
    ['an unknown type', { type: 'pause', id: 2 }],
  ])('rejects %s', (_label, request) => {
    expect(isVadRequest(request)).toBe(false);
  });
});

describe('isVadResponse', () => {
  const stats = { frames: 100, speechFrames: 40, msPerFrame: 0.4, segments: 2, partials: 3 };

  it.each([
    { type: 'ready', id: 1, vad: 'silero', fallbackReason: null },
    { type: 'ready', id: 1, vad: 'energy', fallbackReason: 'model failed to load' },
    { type: 'stats', id: 3, stats },
    { type: 'error', id: 4, message: 'boom' },
  ])('accepts %o', (response) => {
    expect(isVadResponse(response)).toBe(true);
  });

  it.each([
    ['stats with a fractional count', { type: 'stats', id: 3, stats: { ...stats, frames: 1.5 } }],
    ['a ready without a reason field', { type: 'ready', id: 1, vad: 'silero' }],
    ['a non-string error', { type: 'error', id: 4, message: 1 }],
  ])('rejects %s', (_label, response) => {
    expect(isVadResponse(response)).toBe(false);
  });
});

describe('isAudioPortMessage', () => {
  it('accepts flush', () => {
    expect(isAudioPortMessage({ type: 'flush' })).toBe(true);
    expect(isAudioPortMessage({ type: 'flush', id: 1 })).toBe(false);
  });

  it('accepts mono float chunks only', () => {
    expect(isAudioPortMessage({ type: 'audio', samples: new Float32Array(128) })).toBe(true);
    expect(isAudioPortMessage({ type: 'audio', samples: new Float32Array(0) })).toBe(false);
    expect(isAudioPortMessage({ type: 'audio', samples: new Float64Array(128) })).toBe(false);
    expect(isAudioPortMessage({ type: 'audio', samples: new Float32Array(192_001) })).toBe(false);
    expect(
      isAudioPortMessage({ type: 'audio', samples: new Float32Array(128), rate: 48_000 }),
    ).toBe(false);
  });
});

describe('isSegmentEvent', () => {
  const final = {
    type: 'final',
    segmentId: 3,
    startSample: 1_000,
    endSample: 1_100,
    speechEndSample: 1_080,
    reason: 'silence',
    overlapSamples: 0,
    audio: new Float32Array(100),
  };
  const partial = { type: 'partial', segmentId: 3, startSample: 1_000, endSample: 1_050, overlapSamples: 0, audio: new Float32Array(50) };

  it('accepts finals and partials', () => {
    expect(isSegmentEvent(final)).toBe(true);
    expect(isSegmentEvent({ ...final, reason: 'max-length', overlapSamples: 20 })).toBe(true);
    expect(isSegmentEvent(partial)).toBe(true);
    expect(isSegmentEvent({ ...partial, overlapSamples: 50 })).toBe(true);
  });

  it.each([
    ['audio that does not match the sample range', { ...final, audio: new Float32Array(99) }],
    ['an end before the start', { ...final, startSample: 2_000 }],
    ['an unknown reason', { ...final, reason: 'timeout' }],
    ['an overlap longer than the audio', { ...final, overlapSamples: 101 }],
    ['a partial with final-only fields', { ...partial, reason: 'silence' }],
    ['a partial without its overlap', { type: 'partial', segmentId: 3, startSample: 1_000, endSample: 1_050, audio: new Float32Array(50) }],
    ['a partial overlap longer than the audio', { ...partial, overlapSamples: 51 }],
    ['more than one Whisper window', { ...final, endSample: 1_000 + 480_001, audio: new Float32Array(480_001) }],
  ])('rejects %s', (_label, event) => {
    expect(isSegmentEvent(event)).toBe(false);
  });
}); // prettier-ignore

describe('isSegmentPortMessage', () => {
  it('accepts segments and the end-of-stream marker', () => {
    const stats = { frames: 1, speechFrames: 0, msPerFrame: 0.3, segments: 0, partials: 0 };
    expect(isSegmentPortMessage({ type: 'flushed', stats })).toBe(true);
    expect(isSegmentPortMessage({ type: 'flushed' })).toBe(false);
    expect(
      isSegmentPortMessage({
        type: 'partial',
        segmentId: 0,
        startSample: 0,
        endSample: 10,
        overlapSamples: 0,
        audio: new Float32Array(10),
      }),
    ).toBe(true);
  });
});
