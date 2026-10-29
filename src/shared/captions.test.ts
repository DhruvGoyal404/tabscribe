import { describe, expect, it } from 'vitest';
import { planBackend } from '../asr/backend';
import {
  describeStatus,
  isCaptionMessage,
  isCaptionStatus,
  MAX_CAPTION_LENGTH,
  type CaptionStatus,
} from './captions';

const plan = planBackend({
  gpu: { available: false, reason: 'none' },
  webgpuFailure: null,
  crossOriginIsolated: true,
  hardwareConcurrency: 8,
});
const loading: CaptionStatus = {
  state: 'loading',
  plan,
  downloading: true,
  loadedBytes: 40e6,
  totalBytes: 79.7e6,
};

describe('isCaptionStatus', () => {
  it.each<CaptionStatus>([
    loading,
    { state: 'loading', plan, downloading: false, loadedBytes: 0, totalBytes: 0 },
    { state: 'listening', plan, vad: 'silero' },
    { state: 'listening', plan, vad: 'energy' },
    { state: 'error', plan, message: 'ModelNotDownloaded' },
  ])('accepts %o', (status) => {
    expect(isCaptionStatus(status)).toBe(true);
  });

  it.each([
    ['null', null],
    ['an unknown state', { state: 'off', plan }],
    ['a bad plan', { state: 'listening', plan: { ...plan, threads: 0 }, vad: 'silero' }],
    ['negative bytes', { ...loading, loadedBytes: -1 }],
    ['a missing download flag', { state: 'loading', plan, loadedBytes: 0, totalBytes: 0 }],
    ['an unknown VAD', { state: 'listening', plan, vad: 'webrtc' }],
    ['an empty error', { state: 'error', plan, message: '' }],
    ['extra keys', { state: 'listening', plan, vad: 'silero', extra: 1 }],
  ])('rejects %s', (_, value) => {
    expect(isCaptionStatus(value)).toBe(false);
  });
});

describe('isCaptionMessage', () => {
  it('accepts status and caption messages', () => {
    expect(isCaptionMessage({ type: 'captions/status', status: loading })).toBe(true);
    expect(isCaptionMessage({ type: 'captions/caption', caption: { kind: 'final', segmentId: 3, text: '' } })).toBe(true);
    expect(isCaptionMessage({ type: 'captions/caption', caption: { kind: 'partial', segmentId: 0, text: 'He has' } })).toBe(true);
  }); // prettier-ignore

  it.each([
    ['an unknown kind', { type: 'captions/caption', caption: { kind: 'draft', segmentId: 0, text: '' } }],
    ['a negative segment', { type: 'captions/caption', caption: { kind: 'final', segmentId: -1, text: '' } }],
    ['an overlong caption', { type: 'captions/caption', caption: { kind: 'final', segmentId: 0, text: 'x'.repeat(MAX_CAPTION_LENGTH + 1) } }],
    ['a level', { type: 'level', rmsDbfs: -20, peakDbfs: -10, clipped: false }],
  ])('rejects %s', (_, value) => {
    expect(isCaptionMessage(value)).toBe(false);
  }); // prettier-ignore
});

describe('describeStatus', () => {
  it('says what is happening', () => {
    expect(describeStatus(loading)).toBe('Downloading the speech model: 40.0 of 79.7 MB');
    expect(describeStatus({ ...loading, downloading: false })).toBe('Loading the speech model…');
    expect(describeStatus({ state: 'listening', plan, vad: 'silero' })).toBe('Listening for speech');
    expect(describeStatus({ state: 'error', plan, message: 'out of memory' })).toBe('Captions stopped: out of memory');
  }); // prettier-ignore
});
