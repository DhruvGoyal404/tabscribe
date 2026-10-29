import { describe, expect, it } from 'vitest';
import type { SegmentEvent } from '../vad/segmenter';
import {
  CaptionScheduler,
  MAX_MERGED_SAMPLES,
  type Caption,
  type DecodeJob,
  type Decoded,
  type SchedulerOptions,
} from './scheduler';

const final = (
  segmentId: number,
  startSample: number,
  samples: number,
  reason: 'silence' | 'max-length' = 'silence',
  overlapSamples = 0,
): SegmentEvent => ({
  type: 'final',
  segmentId,
  startSample,
  endSample: startSample + samples,
  speechEndSample: startSample + samples,
  reason,
  overlapSamples,
  audio: Float32Array.from({ length: samples }, (_, i) => segmentId * 1000 + i),
});

const partial = (
  segmentId: number,
  startSample: number,
  samples: number,
  overlapSamples = 0,
): SegmentEvent => ({
  type: 'partial',
  segmentId,
  startSample,
  endSample: startSample + samples,
  overlapSamples,
  audio: new Float32Array(samples),
});

const answer = (job: DecodeJob, text: string, overrides: Partial<Decoded> = {}): Decoded => ({
  type: 'decoded',
  id: job.id,
  outcome: 'ok',
  text,
  tokens: [1, 2, 3],
  language: 'en',
  englishProb: 0.99,
  noSpeechProb: 0.01,
  promptTokens: job.prompt?.tokens.length ?? 0,
  attempts: [],
  featuresMs: 1,
  encoderMs: 1,
  ms: 1,
  ...overrides,
});

const WEBGPU: SchedulerOptions = { partials: true, mergeQueued: false, prompting: true };
const WASM: SchedulerOptions = { partials: false, mergeQueued: true, prompting: true };

/** A scheduler with a manual worker: jobs wait until the test answers them. */
function setup(options: SchedulerOptions) {
  const jobs: DecodeJob[] = [];
  const interrupted: number[] = [];
  const captions: Caption[] = [];
  const scheduler = new CaptionScheduler(
    options,
    { decode: (job) => jobs.push(job), interrupt: (id) => interrupted.push(id) },
    (caption) => captions.push(caption),
  );
  const job = (index: number): DecodeJob => {
    const found = jobs[index];
    if (!found) throw new Error(`No job ${String(index)}`);
    return found;
  };
  const finals = () =>
    captions.flatMap((caption) => (caption.type === 'final' ? [caption.text] : []));
  return { scheduler, jobs, job, interrupted, captions, finals };
}

describe('CaptionScheduler', () => {
  it('decodes one job at a time, finals in order', () => {
    const { scheduler, jobs, job, finals } = setup(WEBGPU);
    scheduler.push(final(0, 0, 100));
    scheduler.push(final(1, 200, 100));
    expect(jobs.map((job) => job.segmentIds)).toEqual([[0]]);
    scheduler.onDecoded(answer(job(0), 'zero'));
    expect(jobs.map((job) => job.segmentIds)).toEqual([[0], [1]]);
    scheduler.onDecoded(answer(job(1), 'one'));
    expect(finals()).toEqual(['zero', 'one']);
    expect(scheduler.idle).toBe(true);
  });

  it('ignores an answer for a job that is not running', () => {
    const { scheduler, job, captions } = setup(WEBGPU);
    scheduler.push(final(0, 0, 100));
    scheduler.onDecoded({ ...answer(job(0), 'x'), id: 99 });
    expect(captions).toEqual([]);
  });

  it('shows the newest partial while no final waits', () => {
    const { scheduler, jobs, job, captions } = setup(WEBGPU);
    scheduler.push(partial(0, 0, 100));
    scheduler.push(partial(0, 0, 200));
    scheduler.push(partial(0, 0, 300));
    // The first partial is running; only the newest of the others is kept.
    expect(jobs).toHaveLength(1);
    scheduler.onDecoded(answer(job(0), 'he'));
    expect(jobs).toHaveLength(2);
    expect(jobs[1]?.audio).toHaveLength(300);
    expect(jobs[1]?.kind).toBe('partial');
    scheduler.onDecoded(answer(job(1), 'he has'));
    expect(captions).toEqual([
      { type: 'partial', segmentId: 0, text: 'he' },
      { type: 'partial', segmentId: 0, text: 'he has' },
    ]);
  });

  it('interrupts a running partial when a final arrives, and decodes the final next', () => {
    const { scheduler, jobs, job, interrupted, captions } = setup(WEBGPU);
    scheduler.push(partial(0, 0, 100));
    scheduler.push(final(0, 0, 150));
    expect(interrupted).toEqual([jobs[0]?.id]);
    scheduler.onDecoded(answer(job(0), '', { outcome: 'interrupted', tokens: [] }));
    expect(jobs[1]?.kind).toBe('final');
    scheduler.onDecoded(answer(job(1), 'he has grave doubts'));
    expect(captions.map((caption) => caption.type)).toEqual(['final']);
  });

  it('never shows a partial after its final', () => {
    const { scheduler, jobs, job, captions } = setup(WEBGPU);
    scheduler.push(partial(0, 0, 100));
    scheduler.push(final(0, 0, 150));
    // The partial finished before the interrupt took effect.
    scheduler.onDecoded(answer(job(0), 'he'));
    expect(captions).toEqual([]);
    // A late partial for a finalised segment is ignored too.
    scheduler.push(partial(0, 0, 120));
    scheduler.onDecoded(answer(job(1), 'he has'));
    expect(jobs).toHaveLength(2);
    expect(captions.map((caption) => caption.type)).toEqual(['final']);
  });

  it('skips partials while a final waits, and drops them without the option', () => {
    const gpu = setup(WEBGPU);
    gpu.scheduler.push(final(0, 0, 100));
    gpu.scheduler.push(partial(1, 200, 100));
    gpu.scheduler.push(final(2, 400, 100));
    gpu.scheduler.onDecoded(answer(gpu.job(0), 'a'));
    expect(gpu.jobs[1]?.segmentIds).toEqual([2]);

    const cpu = setup(WASM);
    cpu.scheduler.push(partial(0, 0, 100));
    expect(cpu.jobs).toEqual([]);
    expect(cpu.scheduler.idle).toBe(true);
  });

  it('merges finals that queued behind a busy worker on WASM, without repeating overlaps', () => {
    const { scheduler, job, finals } = setup(WASM);
    scheduler.push(final(0, 0, 100));
    scheduler.push(final(1, 200, 100, 'max-length'));
    scheduler.push(final(2, 280, 100, 'silence', 20));
    scheduler.onDecoded(answer(job(0), 'zero'));
    const merged = job(1);
    expect(merged.segmentIds).toEqual([1, 2]);
    expect(merged.audio).toHaveLength(180);
    expect(merged.audio[100]).toBe(2020); // segment 2's first sample after its overlap
    expect(merged.reason).toBe('silence');
    expect([merged.startSample, merged.endSample]).toEqual([200, 380]);
    scheduler.onDecoded(answer(merged, 'one two'));
    expect(finals()).toEqual(['zero', 'one two']);
  });

  it('merges up to 12 s', () => {
    const { scheduler, jobs, job } = setup(WASM);
    const seconds8 = 8 * 16_000;
    scheduler.push(final(0, 0, 100));
    scheduler.push(final(1, 1_000, seconds8));
    scheduler.push(final(2, 200_000, seconds8));
    scheduler.onDecoded(answer(job(0), 'zero'));
    expect(jobs[1]?.segmentIds).toEqual([1]);
    expect(jobs[1]?.audio.length).toBeLessThanOrEqual(MAX_MERGED_SAMPLES);
  });

  it('prompts after a forced cut and drops the words the overlap repeats', () => {
    const { scheduler, jobs, job, finals } = setup(WEBGPU);
    scheduler.push(final(0, 0, 160_000, 'max-length'));
    scheduler.onDecoded(answer(job(0), 'He has grave doubts whether Sir Frederick'));
    scheduler.push(final(1, 128_000, 100_000, 'silence', 32_000));
    expect(jobs[1]?.prompt).toEqual({ tokens: [1, 2, 3], language: 'en' });
    scheduler.onDecoded(answer(job(1), 'Sir Frederick Leighton’s work is Greek.'));
    expect(finals()).toEqual([
      'He has grave doubts whether Sir Frederick',
      'Leighton’s work is Greek.',
    ]);
  });

  it('drops the words the overlap repeats from partials too', () => {
    const { scheduler, job, captions } = setup({ ...WEBGPU, prompting: false });
    scheduler.push(final(0, 0, 160_000, 'max-length'));
    scheduler.onDecoded(answer(job(0), '...just the ritual of cracking something open.'));
    scheduler.push(partial(1, 128_000, 48_000, 32_000));
    scheduler.onDecoded(answer(job(1), 'of cracking something open and sharing it'));
    expect(captions.at(-1)).toEqual({ type: 'partial', segmentId: 1, text: 'and sharing it' });
    // A partial after a natural pause has no overlap, so nothing is dropped.
    scheduler.push(final(1, 128_000, 64_000, 'silence', 32_000));
    scheduler.onDecoded(answer(job(2), 'of cracking something open and sharing it with friends'));
    scheduler.push(partial(2, 200_000, 32_000));
    scheduler.onDecoded(answer(job(3), 'with friends we'));
    expect(captions.at(-1)).toEqual({ type: 'partial', segmentId: 2, text: 'with friends we' });
  });

  it('does not prompt when prompting is off, or after a dropped decode', () => {
    const off = setup({ ...WEBGPU, prompting: false });
    off.scheduler.push(final(0, 0, 160_000, 'max-length'));
    off.scheduler.onDecoded(answer(off.job(0), 'one two three'));
    off.scheduler.push(final(1, 128_000, 100_000, 'silence', 32_000));
    expect(off.jobs[1]?.prompt).toBeNull();

    const dropped = setup(WEBGPU);
    dropped.scheduler.push(final(0, 0, 160_000, 'max-length'));
    dropped.scheduler.onDecoded(answer(dropped.job(0), '', { outcome: 'loop', tokens: [] }));
    dropped.scheduler.push(final(1, 128_000, 100_000, 'silence', 32_000));
    expect(dropped.jobs[1]?.prompt).toBeNull();
    dropped.scheduler.onDecoded(answer(dropped.job(1), 'Sir Frederick'));
    expect(dropped.finals()).toEqual(['', 'Sir Frederick']);
  });
});
