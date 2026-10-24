// Decodes a fixture's VAD finals through the caption scheduler, one at a time and in order, as the
// captions would show them. Offline, so it measures text quality (WER, loops) rather than latency
// (piece 4's live mode does that); no partials and no merging, so each final is one decode.
import { hasLoop } from '../asr/guards';
import type { DecodeOutcome, DecodePrompt } from '../asr/protocol';
import { CaptionScheduler, type Caption, type Decoded } from '../asr/scheduler';
import type { SegmentEvent } from '../vad/segmenter';

export type FinalSegment = Extract<SegmentEvent, { type: 'final' }>;
export type DecodeFn = (audio: Float32Array, prompt: DecodePrompt | null) => Promise<Decoded>;

export interface StreamStats {
  segments: number;
  forcedCuts: number;
  /** Segments decoded with a prompt the worker used. */
  prompted: number;
  /** Attempts the loop guard caught (each one was retried or dropped). */
  loopsCaught: number;
  /** Shown captions that still contain a repeated phrase: should stay 0. */
  loopsShown: number;
  /** Attempts after the first, across segments. */
  retries: number;
  outcomes: Partial<Record<DecodeOutcome, number>>;
  /** Sum of the worker's per-segment time. */
  decodeMs: number;
}

export interface StreamRun {
  captions: Extract<Caption, { type: 'final' }>[];
  transcript: string;
  stats: StreamStats;
}

export function decodeStream(
  finals: readonly FinalSegment[],
  decode: DecodeFn,
  prompting: boolean,
): Promise<StreamRun> {
  return new Promise((resolve, reject) => {
    const captions: StreamRun['captions'] = [];
    const stats: StreamStats = {
      segments: finals.length,
      forcedCuts: finals.filter((segment) => segment.reason === 'max-length').length,
      prompted: 0,
      loopsCaught: 0,
      loopsShown: 0,
      retries: 0,
      outcomes: {},
      decodeMs: 0,
    };
    const finish = () => {
      const transcript = captions
        .map((caption) => caption.text)
        .filter(Boolean)
        .join(' ');
      resolve({ captions, transcript, stats });
    };
    const scheduler: CaptionScheduler = new CaptionScheduler(
      { partials: false, mergeQueued: false, prompting },
      {
        decode: (job) => {
          decode(job.audio, job.prompt).then((result) => {
            scheduler.onDecoded({ ...result, id: job.id });
          }, reject);
        },
        interrupt: () => undefined,
      },
      (caption) => {
        if (caption.type !== 'final') return;
        const { result } = caption;
        captions.push(caption);
        stats.outcomes[result.outcome] = (stats.outcomes[result.outcome] ?? 0) + 1;
        if (result.promptTokens > 0) stats.prompted += 1;
        stats.loopsCaught += result.attempts.filter((attempt) => attempt.verdict === 'loop').length;
        stats.retries += Math.max(0, result.attempts.length - 1);
        if (caption.text && hasLoop(caption.text)) stats.loopsShown += 1;
        stats.decodeMs += result.ms;
        if (captions.length === finals.length) finish();
      },
    );
    for (const segment of finals) scheduler.push(segment);
    if (finals.length === 0) finish();
  });
}
