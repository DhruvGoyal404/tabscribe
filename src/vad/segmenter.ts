// Turns a stream of 16 kHz VAD frames (audio plus speech probability) into speech segments for the
// ASR worker. Pure and synchronous, so it can be unit-tested without a model.
//
// Hysteresis as in Silero's get_speech_timestamps: speech starts when the probability reaches
// `startThreshold`, and a pause starts when it falls below `endThreshold`; values in between keep
// whichever state is current. A segment ends after `minSilenceMs` of pause, padded by `padMs` on
// both sides. Long speech without a pause is cut when it reaches `maxSegmentMs`, after the quietest
// frame of its last `cutSearchMs` (as Silero's get_speech_timestamps splits at its last silence),
// so the cut tends to fall between words. The next segment repeats the last `overlapMs` before the
// cut, so a word the cut splits anyway is heard whole once (the ASR side de-duplicates).

export interface SegmenterOptions {
  sampleRate: number;
  startThreshold: number;
  endThreshold: number;
  /** Shorter bursts are dropped (clicks, coughs). */
  minSpeechMs: number;
  /** Pause length that ends an utterance. Part of the caption latency. */
  minSilenceMs: number;
  /** Audio kept before speech starts and after it ends. */
  padMs: number;
  /** Forced cut for speech without pauses. Bounds the worst-case latency. */
  maxSegmentMs: number;
  /** A forced cut goes after the quietest frame of this last stretch of the segment; 0 cuts at the maximum. */
  cutSearchMs: number;
  /** Repeated at the start of the segment that follows a forced cut. */
  overlapMs: number;
  /** First partial snapshot of a growing segment, measured from speech start. */
  firstPartialMs: number;
  partialIntervalMs: number;
}

export const DEFAULT_SEGMENTER: SegmenterOptions = {
  sampleRate: 16_000,
  startThreshold: 0.5,
  endThreshold: 0.35,
  minSpeechMs: 250,
  minSilenceMs: 500,
  padMs: 200,
  maxSegmentMs: 10_000,
  cutSearchMs: 2_000,
  overlapMs: 2_000,
  firstPartialMs: 1_000,
  partialIntervalMs: 2_000,
};

export type EndReason = 'silence' | 'max-length' | 'flush';

/**
 * Sample positions are absolute, counted from the start of the stream at `sampleRate`. Segment ids
 * increase by one per segment; a dropped burst leaves a gap. Partials share their segment's id.
 */
export type SegmentEvent =
  | {
      type: 'partial';
      segmentId: number;
      startSample: number;
      endSample: number;
      /** Leading samples repeated from the previous segment after a forced cut. */
      overlapSamples: number;
      audio: Float32Array;
    }
  | {
      type: 'final';
      segmentId: number;
      startSample: number;
      endSample: number;
      /** Where speech stopped (before padding): the utterance end latency is measured from. */
      speechEndSample: number;
      reason: EndReason;
      /** Leading samples repeated from the previous segment after a forced cut. */
      overlapSamples: number;
      audio: Float32Array;
    };

function concat(frames: readonly Float32Array[]): Float32Array {
  const out = new Float32Array(frames.reduce((sum, frame) => sum + frame.length, 0));
  let offset = 0;
  for (const frame of frames) {
    out.set(frame, offset);
    offset += frame.length;
  }
  return out;
}

/** The newest frames covering at least `samples` samples (whole frames). */
function tail(frames: readonly Float32Array[], samples: number): Float32Array[] {
  const kept: Float32Array[] = [];
  let total = 0;
  for (let i = frames.length - 1; i >= 0 && total < samples; i--) {
    const frame = frames[i];
    if (!frame) break;
    kept.unshift(frame);
    total += frame.length;
  }
  return kept;
}

const length = (frames: readonly Float32Array[]): number =>
  frames.reduce((sum, frame) => sum + frame.length, 0);

const meanSquare = (frame: Float32Array): number =>
  frame.reduce((sum, sample) => sum + sample * sample, 0) / Math.max(1, frame.length);

/** Index of the quietest of the frames covering the last `samples`; the latest one on a tie. */
function quietest(frames: readonly Float32Array[], samples: number): number {
  let best = frames.length - 1;
  let bestLevel = Infinity;
  let covered = 0;
  for (let i = frames.length - 1; i >= 0 && covered < samples; i--) {
    const frame = frames[i];
    if (!frame) break;
    const level = meanSquare(frame);
    if (level < bestLevel) {
      best = i;
      bestLevel = level;
    }
    covered += frame.length;
  }
  return best;
}

export class Segmenter {
  private readonly options: SegmenterOptions;
  private readonly samples: (ms: number) => number;
  /** Samples consumed so far. */
  private position = 0;
  private nextId = 0;

  /** While idle: the last `padMs` of audio, prepended when speech starts. */
  private preRoll: Float32Array[] = [];

  /** While in speech: */
  private inSpeech = false;
  private frames: Float32Array[] = [];
  private segmentStart = 0;
  private speechStart = 0;
  private pauseStart: number | null = null;
  private overlap = 0;
  private nextPartial = 0;

  constructor(options: Partial<SegmenterOptions> = {}) {
    this.options = { ...DEFAULT_SEGMENTER, ...options };
    const { endThreshold, startThreshold } = this.options;
    if (!(endThreshold <= startThreshold)) throw new RangeError('endThreshold > startThreshold');
    const rate = this.options.sampleRate;
    this.samples = (ms) => Math.round((ms * rate) / 1000);
  }

  /** Feeds one VAD frame. Returns the events it completes (often none). */
  push(frame: Float32Array, probability: number): SegmentEvent[] {
    const frameStart = this.position;
    const frameEnd = frameStart + frame.length;
    this.position = frameEnd;
    const o = this.options;

    if (!this.inSpeech) {
      this.preRoll.push(frame);
      this.preRoll = tail(this.preRoll, this.samples(o.padMs) + frame.length);
      if (probability < o.startThreshold) return [];
      // Speech starts at this frame; keep up to padMs before it.
      const before = tail(this.preRoll.slice(0, -1), this.samples(o.padMs));
      this.frames = [...before, frame];
      this.segmentStart = frameStart - length(before);
      this.speechStart = frameStart;
      this.startSegment(frameStart);
      this.preRoll = [];
      this.inSpeech = true;
      return [];
    }

    this.frames.push(frame);
    if (probability >= o.startThreshold) this.pauseStart = null;
    else if (probability < o.endThreshold) this.pauseStart ??= frameStart;

    if (this.pauseStart !== null && frameEnd - this.pauseStart >= this.samples(o.minSilenceMs)) {
      return this.endOnPause(this.pauseStart, frameEnd);
    }
    if (frameEnd - this.segmentStart >= this.samples(o.maxSegmentMs)) {
      return [this.cut()];
    }
    if (frameEnd >= this.nextPartial) {
      this.nextPartial += this.samples(o.partialIntervalMs);
      return [
        {
          type: 'partial',
          segmentId: this.nextId,
          startSample: this.segmentStart,
          endSample: frameEnd,
          overlapSamples: this.overlap,
          audio: concat(this.frames),
        },
      ];
    }
    return [];
  }

  /** Ends any segment in progress (capture stopped). */
  flush(): SegmentEvent[] {
    if (!this.inSpeech) return [];
    const speechEnd = this.pauseStart ?? this.position;
    this.inSpeech = false;
    const events = this.final(speechEnd, this.position, 'flush');
    this.frames = [];
    this.preRoll = [];
    return events;
  }

  private startSegment(speechStart: number): void {
    this.speechStart = speechStart;
    this.pauseStart = null;
    this.nextPartial = speechStart + this.samples(this.options.firstPartialMs);
  }

  private endOnPause(speechEnd: number, frameEnd: number): SegmentEvent[] {
    const end = Math.min(frameEnd, speechEnd + this.samples(this.options.padMs));
    const events = this.final(speechEnd, end, 'silence');
    // The pause after the padding becomes pre-roll for the next utterance.
    this.preRoll = tail(this.frames, this.samples(this.options.padMs));
    this.frames = [];
    this.inSpeech = false;
    this.overlap = 0;
    return events;
  }

  private cut(): SegmentEvent {
    const kept = quietest(this.frames, this.samples(this.options.cutSearchMs)) + 1;
    const after = this.frames.slice(kept);
    this.frames = this.frames.slice(0, kept);
    const cutAt = this.segmentStart + length(this.frames);
    const pause = this.pauseStart;
    // Always emitted: speech has been going for maxSegmentMs.
    const event = this.emit(pause !== null && pause < cutAt ? pause : cutAt, cutAt, 'max-length');
    const carried = tail(this.frames, this.samples(this.options.overlapMs));
    this.frames = [...carried, ...after];
    this.overlap = length(carried);
    this.segmentStart = cutAt - this.overlap;
    this.startSegment(this.segmentStart);
    // A pause that began after the cut belongs to the new segment.
    if (pause !== null && pause >= cutAt) this.pauseStart = pause;
    return event;
  }

  /** Emits the segment unless it is too short to be speech. */
  private final(speechEnd: number, end: number, reason: EndReason): SegmentEvent[] {
    if (speechEnd - this.speechStart < this.samples(this.options.minSpeechMs)) {
      this.nextId++;
      return [];
    }
    return [this.emit(speechEnd, end, reason)];
  }

  private emit(speechEnd: number, end: number, reason: EndReason): SegmentEvent {
    const audio = concat(this.frames).slice(0, end - this.segmentStart);
    return {
      type: 'final',
      segmentId: this.nextId++,
      startSample: this.segmentStart,
      endSample: this.segmentStart + audio.length,
      speechEndSample: speechEnd,
      reason,
      overlapSamples: this.overlap,
      audio,
    };
  }
}
