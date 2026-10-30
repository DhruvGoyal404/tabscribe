import { describe, expect, it } from 'vitest';
import { planBackend } from '../asr/backend';
import { applyCaption, applyStatus, EMPTY } from './text';

const plan = planBackend({
  gpu: { available: false, reason: 'none' },
  webgpuFailure: null,
  crossOriginIsolated: true,
  hardwareConcurrency: 8,
});

describe('overlay text', () => {
  it('shows the newest final and the partial after it', () => {
    let text = applyCaption(EMPTY, { kind: 'partial', segmentId: 0, text: 'He has' });
    expect(text).toEqual({ final: '', partial: 'He has' });
    text = applyCaption(text, { kind: 'final', segmentId: 0, text: 'He has grave doubts.' });
    expect(text).toEqual({ final: 'He has grave doubts.', partial: '' });
    text = applyCaption(text, { kind: 'partial', segmentId: 1, text: 'Whether' });
    expect(text).toEqual({ final: 'He has grave doubts.', partial: 'Whether' });
  });

  it('keeps the last final when a decode is dropped, but ends its partial', () => {
    const text = { final: 'He has grave doubts.', partial: 'thank you' };
    expect(applyCaption(text, { kind: 'final', segmentId: 1, text: '' })).toEqual({
      final: 'He has grave doubts.',
      partial: '',
    });
  });

  it('clears unless the pipeline is listening', () => {
    const text = { final: 'x', partial: 'y' };
    expect(applyStatus(text, { state: 'listening', plan, vad: 'silero' })).toBe(text);
    expect(applyStatus(text, { state: 'error', plan, message: 'out of memory' })).toEqual(EMPTY);
  });
});
