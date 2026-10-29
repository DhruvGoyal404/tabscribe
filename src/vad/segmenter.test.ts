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
  // Cuts at the forced point (no search), since these samples get louder with their position.
  cutSearchMs: 0,
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

/** Speech at a given loudness per 100 ms frame (0 is a gap between words), then a pause. */
function runLevels(levels: number[], overrides: Partial<SegmenterOptions> = {}): SegmentEvent[] {
  const segmenter = new Segmenter({ ...options, cutSearchMs: 2_000, ...overrides });
  const events: SegmentEvent[] = [];
  [...levels, ...Array<number>(10).fill(0)].forEach((level, i) => {
    const frame = new Float32Array(FRAME).fill(level);
    events.push(...segmenter.push(frame, i < levels.length ? 0.9 : 0));
  });
  return [...events, ...segmenter.flush()];
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

  it('makes a forced cut after the quietest frame of its last 2 s', () => {
    // 20 s of speech with a gap between words at 8.5 s, and at 16.0 s.
    const levels = Array.from({ length: 200 }, (_, i) => (i === 85 || i === 160 ? 0.01 : 0.5));
    const events = finals(runLevels(levels));
    expect(events.map((e) => [e.reason, e.startSample, e.endSample, e.overlapSamples])).toEqual([
      ['max-length', 0, 860, 0], // the gap at 8.5–8.6 s ends the segment
      ['max-length', 660, 1610, 200], // 2 s back from the cut, then to the gap at 16.0–16.1 s
      ['silence', 1410, 2020, 200],
    ]);
    // Each segment is the stream from its start: the overlap, then what followed the cut.
    expect(events[1]?.audio.length).toBe(1610 - 660);
  });

  it('cuts at the forced point when no frame is quieter', () => {
    const events = finals(runLevels(Array<number>(150).fill(0.5)));
    expect(events.map((e) => [e.startSample, e.endSample])).toEqual([
      [0, 1000],
      [800, 1520],
    ]);
  });

  it('keeps a pause that starts after the cut in the next segment', () => {
    // The quietest frame is at 8.5 s; a pause starts at 9.6 s, after the cut but before 10 s.
    const segmenter = new Segmenter({ ...options, cutSearchMs: 2_000 });
    const events: SegmentEvent[] = [];
    for (let i = 0; i < 110; i++) {
      const frame = new Float32Array(FRAME).fill(i === 85 ? 0.01 : 0.5);
      events.push(...segmenter.push(frame, i >= 96 ? 0.1 : 0.9));
    }
    expect(finals(events).map((e) => [e.reason, e.endSample, e.speechEndSample])).toEqual([
      ['max-length', 860, 860],
      ['silence', 980, 960], // the pause at 9.6 s ended the next segment after 500 ms
    ]);
  });

  it('marks the overlap on the partials after a forced cut too', () => {
    const snapshots = partials(run([...seconds(14, 0.9), ...seconds(1, 0)]));
    expect(snapshots.map((e) => [e.segmentId, e.startSample, e.overlapSamples])).toEqual([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [1, 800, 200],
      [1, 800, 200],
      [1, 800, 200],
    ]);
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
