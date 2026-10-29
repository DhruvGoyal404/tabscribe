// Decides what the ASR worker decodes next from the VAD's segment events, and turns its results
// into captions. Pure: the worker is behind a two-method port, so this runs the same in the
// offscreen document, the bench and the tests.
//
// - Finals go first and in order. The worker decodes one request at a time, so captions come back
//   in segment order.
// - A partial (a preview of a growing segment) is decoded only when no final is waiting, and only
//   the newest one per segment. A final that arrives while a partial runs interrupts it, so the
//   final waits at most one decoder step. A partial's caption is never shown after its final.
// - On WASM the encoder costs seconds per call whatever the length, so finals that queued behind a
//   busy worker are decoded as one call (up to 12 s), and there are no partials.
// - Overlap repeats are dropped, from partials as well as finals, and the restricted previous-text
//   prompt is applied (context.ts).
import { ASR_SAMPLE_RATE } from './resample';
import { dropRepeatedPrefix, PromptContext } from './context';
import type { DecodeKind, DecodePrompt, WorkerResponse } from './protocol';
import type { EndReason, SegmentEvent } from '../vad/segmenter';

export type Decoded = Extract<WorkerResponse, { type: 'decoded' }>;
type Final = Extract<SegmentEvent, { type: 'final' }>;
type Partial = Extract<SegmentEvent, { type: 'partial' }>;

/** Longest merged decode: well inside Whisper's 30 s window and the WASM latency budget. */
export const MAX_MERGED_SAMPLES = 12 * ASR_SAMPLE_RATE;

export interface SchedulerOptions {
  /** Preview growing segments (WebGPU). */
  partials: boolean;
  /** Decode queued finals as one call (WASM). */
  mergeQueued: boolean;
  /** The restricted previous-text prompt. Off for captions for now; the bench measures both. */
  prompting: boolean;
}

export interface DecodeJob {
  id: number;
  kind: DecodeKind;
  audio: Float32Array;
  prompt: DecodePrompt | null;
  /** Segments this job covers: one, or several merged finals. */
  segmentIds: number[];
  startSample: number;
  endSample: number;
  /** Of the last segment: the prompt policy looks at how it ended. */
  reason: EndReason | null;
  /** Leading samples repeated from the segment before (the first merged segment's). */
  overlapSamples: number;
}

export type Caption =
  | { type: 'partial'; segmentId: number; text: string }
  | {
      type: 'final';
      segmentIds: number[];
      startSample: number;
      endSample: number;
      /** Empty when the decode was dropped (no speech, loop, not English). */
      text: string;
      result: Decoded;
    };

export interface AsrPort {
  decode(job: DecodeJob): void;
  interrupt(id: number): void;
}

const shows = (result: Decoded): boolean =>
  result.outcome === 'ok' || result.outcome === 'low-confidence';

function concat(parts: readonly Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

export class CaptionScheduler {
  private readonly finals: Final[] = [];
  private partial: Partial | null = null;
  private running: DecodeJob | null = null;
  private readonly finalised = new Set<number>();
  private readonly context = new PromptContext();
  /** The previous final's text, if it was shown: what an overlap can repeat. */
  private previousText = '';
  private nextId = 0;

  constructor(
    private readonly options: SchedulerOptions,
    private readonly asr: AsrPort,
    private readonly onCaption: (caption: Caption) => void,
  ) {}

  push(event: SegmentEvent): void {
    if (event.type === 'final') {
      this.finals.push(event);
      this.finalised.add(event.segmentId);
      if (this.partial?.segmentId === event.segmentId) this.partial = null;
      if (this.running?.kind === 'partial') this.asr.interrupt(this.running.id);
    } else if (this.options.partials && !this.finalised.has(event.segmentId)) {
      this.partial = event;
    }
    this.pump();
  }

  /** The worker's answer to the running job. */
  onDecoded(result: Decoded): void {
    const job = this.running;
    if (job?.id !== result.id) return;
    this.running = null;
    if (job.kind === 'final') this.finish(job, result);
    else {
      const segmentId = job.segmentIds[0] ?? -1;
      if (shows(result) && !this.finalised.has(segmentId)) {
        this.onCaption({ type: 'partial', segmentId, text: this.newText(job, result) });
      }
    }
    this.pump();
  }

  /** Nothing queued or running. */
  get idle(): boolean {
    return !this.running && this.finals.length === 0 && !this.partial;
  }

  /** The result's text without the words its overlap repeats from the previous final. */
  private newText(job: DecodeJob, result: Decoded): string {
    return job.overlapSamples > 0
      ? dropRepeatedPrefix(this.previousText, result.text)
      : result.text;
  }

  private finish(job: DecodeJob, result: Decoded): void {
    const text = shows(result) ? this.newText(job, result) : '';
    this.onCaption({
      type: 'final',
      segmentIds: job.segmentIds,
      startSample: job.startSample,
      endSample: job.endSample,
      text,
      result,
    });
    if (result.outcome === 'interrupted' || job.reason === null) this.context.reset();
    else {
      this.context.record({
        startSample: job.startSample,
        endSample: job.endSample,
        reason: job.reason,
        verdict: result.outcome,
        language: result.language,
        tokens: result.tokens,
      });
    }
    this.previousText = shows(result) ? result.text : '';
  }

  private promptFor(startSample: number): DecodePrompt | null {
    return this.options.prompting ? this.context.promptFor(startSample) : null;
  }

  private pump(): void {
    if (this.running) return;
    const job = this.finals.length > 0 ? this.finalJob() : this.partialJob();
    if (!job) return;
    this.running = job;
    this.asr.decode(job);
  }

  private finalJob(): DecodeJob {
    const taken: Final[] = [];
    let samples = 0;
    while (this.finals.length > 0) {
      const next = this.finals[0];
      if (!next) break;
      // Later merged segments drop their overlap: it's the end of the one before, already in.
      const length = next.audio.length - (taken.length > 0 ? next.overlapSamples : 0);
      if (
        taken.length > 0 &&
        (!this.options.mergeQueued || samples + length > MAX_MERGED_SAMPLES)
      ) {
        break;
      }
      taken.push(next);
      this.finals.shift();
      samples += length;
    }
    const first = taken[0];
    const last = taken.at(-1);
    if (!first || !last) throw new Error('No final to decode');
    const audio = concat(
      taken.map((segment, i) =>
        i === 0 ? segment.audio : segment.audio.subarray(segment.overlapSamples),
      ),
    );
    return {
      id: this.nextId++,
      kind: 'final',
      audio,
      prompt: this.promptFor(first.startSample),
      segmentIds: taken.map((segment) => segment.segmentId),
      startSample: first.startSample,
      endSample: last.endSample,
      reason: last.reason,
      overlapSamples: first.overlapSamples,
    };
  }

  private partialJob(): DecodeJob | null {
    const partial = this.partial;
    if (!partial) return null;
    this.partial = null;
    return {
      id: this.nextId++,
      kind: 'partial',
      audio: partial.audio,
      prompt: this.promptFor(partial.startSample),
      segmentIds: [partial.segmentId],
      startSample: partial.startSample,
      endSample: partial.endSample,
      reason: null,
      overlapSamples: partial.overlapSamples,
    };
  }
}
