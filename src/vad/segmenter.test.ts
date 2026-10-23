import { describe, expect, it } from 'vitest';
import { Segmenter, type SegmentEvent, type SegmenterOptions } from './segmenter';

// 100 samples per "second" keeps the numbers readable: a 10-sample frame is 100 ms.
const RATE = 100;
const FRAME = 10;
const options: Partial<SegmenterOptions> = {
  sampleRate: RATE,
  minSpeechMs: 250,
  minSilenceMs: 500,
  padMs: 200,
  maxSegmentMs: 10_000,
  overlapMs: 2_000,
  firstPartialMs: 1_000,
  partialIntervalMs: 2_000,
};

/** Runs a probability per 100 ms frame; each sample holds its own stream position. */
function run(probabilities: number[], overrides: Partial<SegmenterOptions> = {}): SegmentEvent[] {
  const segmenter = new Segmenter({ ...options, ...overrides });
  const events: SegmentEvent[] = [];
  probabilities.forEach((probability, i) => {
    const frame = Float32Array.from({ length: FRAME }, (_, j) => i * FRAME + j);
    events.push(...segmenter.push(frame, probability));
  });
  events.push(...segmenter.flush());
  return events;
}

const finals = (events: SegmentEvent[]) => events.filter((e) => e.type === 'final');
const partials = (events: SegmentEvent[]) => events.filter((e) => e.type === 'partial');
const seconds = (n: number, p: number) => Array<number>(n * 10).fill(p);

describe('Segmenter', () => {
  it('cuts one utterance at the pause, padded on both sides', () => {
    // 1 s silence, 3 s speech, 1 s silence.
    const [segment, ...rest] = finals(
      run([...seconds(1, 0), ...seconds(3, 0.9), ...seconds(1, 0)]),
    );
    expect(rest).toHaveLength(0);
    expect(segment).toMatchObject({
      reason: 'silence',
      startSample: 80, // speech at 100, minus 200 ms
      speechEndSample: 400,
      endSample: 420, // plus 200 ms of the pause
      overlapSamples: 0,
    });
    // The audio is exactly the stream between start and end.
    expect(segment?.audio[0]).toBe(80);
    expect(segment?.audio.at(-1)).toBe(419);
    expect(segment?.audio).toHaveLength(340);
  });

  it('keeps short dips inside one utterance', () => {
    // A 300 ms dip is shorter than the 500 ms pause that ends an utterance.
    const events = run([...seconds(2, 0.9), 0.1, 0.1, 0.1, ...seconds(2, 0.9), ...seconds(1, 0)]);
    expect(finals(events)).toHaveLength(1);
  });

  it('splits at pauses of at least minSilenceMs', () => {
    const events = run([
      ...seconds(2, 0.9),
      ...seconds(0.5, 0),
      ...seconds(2, 0.9),
      ...seconds(1, 0),
    ]);
    expect(finals(events).map((e) => e.speechEndSample)).toEqual([200, 450]);
  });

  it('treats probabilities between the thresholds as "no change"', () => {
    // 0.4 neither starts speech (< 0.5) nor starts a pause (>= 0.35).
    expect(finals(run([...seconds(2, 0.4), ...seconds(1, 0)]))).toHaveLength(0);
    // ...and doesn't stop a pause from ending the utterance either.
    const events = run([...seconds(2, 0.9), 0.2, ...seconds(1, 0.4), ...seconds(2, 0.9)]);
    // The second utterance runs to the end of the input and is flushed there.
    expect(finals(events).map((e) => e.speechEndSample)).toEqual([200, 510]);
  });

  it('drops bursts shorter than minSpeechMs, leaving an id gap', () => {
    const events = run([
      ...seconds(1, 0),
      0.9,
      0.9,
      ...seconds(1, 0),
      ...seconds(1, 0.9),
      ...seconds(1, 0),
    ]);
    const kept = finals(events);
    expect(kept).toHaveLength(1);
    expect(kept[0]?.segmentId).toBe(1);
  });

  it('force-cuts long speech and repeats the overlap in the next segment', () => {
    const events = finals(run([...seconds(25, 0.9), ...seconds(1, 0)]));
    expect(events.map((e) => [e.reason, e.startSample, e.endSample, e.overlapSamples])).toEqual([
      ['max-length', 0, 1000, 0],
      ['max-length', 800, 1800, 200],
      ['silence', 1600, 2520, 200],
    ]);
    // The overlap is the same audio, not new samples.
    expect(events[1]?.audio.subarray(0, 200)).toEqual(events[0]?.audio.subarray(800, 1000));
  });

  it('emits growing partials: first at 1 s, then every 2 s', () => {
    const events = run([...seconds(6, 0.9), ...seconds(1, 0)]);
    const snapshots = partials(events);
    expect(snapshots.map((e) => e.endSample)).toEqual([100, 300, 500]);
    const [final] = finals(events);
    for (const partial of snapshots) {
      expect(partial.segmentId).toBe(final?.segmentId);
      expect(final?.audio.subarray(0, partial.audio.length)).toEqual(partial.audio);
    }
  });

  it('flushes speech still in progress', () => {
    const [segment] = finals(run(seconds(2, 0.9)));
    expect(segment).toMatchObject({ reason: 'flush', startSample: 0, endSample: 200 });
  });

  it('uses whole frames of pre-roll at the very start of the stream', () => {
    const [segment] = finals(run([0.9, ...seconds(1, 0.9), ...seconds(1, 0)]));
    expect(segment?.startSample).toBe(0);
  });

  it('rejects an end threshold above the start threshold', () => {
    expect(() => new Segmenter({ startThreshold: 0.3, endThreshold: 0.5 })).toThrow(RangeError);
  });
});
