// Live mode: a fixture played at real-time pace through the VAD worker and the caption scheduler,
// with the scheduler options the extension uses (partials on WebGPU, merged finals on WASM),
// timing how long each caption takes to appear after the speech it shows.
import { ASR_SAMPLE_RATE } from '../asr/resample';
import {
  CaptionScheduler,
  type Caption,
  type DecodeJob,
  type Decoded,
  type SchedulerOptions,
} from '../asr/scheduler';
import { isSegmentPortMessage, isVadResponse, type VadKind } from '../vad/protocol';
import type { SegmentEvent } from '../vad/segmenter';
import { CHUNK } from './vad-run';

/** The ASR worker, as the live run drives it: one decode at a time, as the scheduler asks. */
export interface LiveAsr {
  decode(job: DecodeJob): Promise<Decoded>;
  interrupt(jobId: number): void;
}

/**
 * Caption delays against the audio clock. A final is timed from where its speech ended (the start of
 * the pause, or the forced cut), a partial from the end of the audio it previews. Only captions
 * with text count: those are what the viewer sees.
 */
export class LatencyLog {
  readonly finalMs: number[] = [];
  readonly partialMs: number[] = [];
  private readonly speechEnd = new Map<number, number>();

  /** `startMs`: when the stream's first sample played. */
  constructor(private readonly startMs: number) {}

  /** When the stream had played up to `sample` (16 kHz). */
  /** Finals the VAD cut, shown or not. */
  get segments(): number {
    return this.speechEnd.size;
  }

  playedAt(sample: number): number {
    return this.startMs + (sample * 1000) / ASR_SAMPLE_RATE;
  }

  segment(event: SegmentEvent): void {
    if (event.type === 'final') this.speechEnd.set(event.segmentId, event.speechEndSample);
  }

  /** A caption shown at `nowMs`, decoded from `job`. */
  shown(caption: Caption, job: DecodeJob, nowMs: number): void {
    if (!caption.text) return;
    if (caption.type === 'partial') {
      this.partialMs.push(nowMs - this.playedAt(job.endSample));
      return;
    }
    const end = this.speechEnd.get(caption.segmentIds.at(-1) ?? -1);
    if (end !== undefined) this.finalMs.push(nowMs - this.playedAt(end));
  }
}

export interface LiveRun {
  vad: VadKind;
  log: LatencyLog;
  /** The finals' text, joined. */
  transcript: string;
  /** Wall-clock time from the first sample until every caption was out. */
  ms: number;
}

/** Plays `capture` (mono at `inputRate`) in real time; resolves once every caption is out. */
export async function runLive(
  capture: Float32Array,
  inputRate: number,
  options: SchedulerOptions,
  asr: LiveAsr,
): Promise<LiveRun> {
  const worker = new Worker(new URL('../vad/vad.worker.ts', import.meta.url), { type: 'module' });
  const audio = new MessageChannel();
  const segments = new MessageChannel();
  try {
    const vad = await new Promise<VadKind>((resolve, reject) => {
      worker.onmessage = (event: MessageEvent<unknown>) => {
        const response = event.data;
        if (!isVadResponse(response)) return;
        if (response.type === 'ready') resolve(response.vad);
        else if (response.type === 'error') reject(new Error(response.message));
      };
      worker.onerror = (event) => {
        reject(new Error(event.message));
      };
      worker.postMessage(
        {
          type: 'init',
          id: 0,
          inputRate,
          vad: 'silero',
          tuning: {},
          audio: audio.port2,
          segments: segments.port1,
        },
        [audio.port2, segments.port1],
      );
    });

    const startMs = performance.now();
    const log = new LatencyLog(startMs);
    const finals: string[] = [];
    let flushed = false;

    const result = new Promise<void>((resolve, reject) => {
      let decoding: DecodeJob | null = null;
      const settle = () => {
        if (flushed && scheduler.idle) resolve();
      };
      const scheduler: CaptionScheduler = new CaptionScheduler(
        options,
        {
          decode: (job) => {
            asr.decode(job).then((decoded) => {
              decoding = job;
              scheduler.onDecoded({ ...decoded, id: job.id });
              settle();
            }, reject);
          },
          interrupt: (id) => {
            asr.interrupt(id);
          },
        },
        (caption) => {
          const now = performance.now();
          if (decoding) log.shown(caption, decoding, now);
          if (caption.type === 'final' && caption.text) finals.push(caption.text);
        },
      );
      segments.port2.onmessage = (event: MessageEvent<unknown>) => {
        const message = event.data;
        if (!isSegmentPortMessage(message)) return;
        if (message.type === 'flushed') {
          flushed = true;
          settle();
          return;
        }
        log.segment(message);
        scheduler.push(message);
      };
    });

    // Post each chunk once its last sample has "played", as the capture worklet would.
    let posted = 0;
    await new Promise<void>((resolve) => {
      const tick = () => {
        const due = Math.min(capture.length, ((performance.now() - startMs) / 1000) * inputRate);
        while (posted < capture.length && (posted + CHUNK <= due || due >= capture.length)) {
          const samples = capture.slice(posted, posted + CHUNK);
          // Counted before posting: the transfer detaches the buffer, so its length reads 0 after.
          posted += samples.length;
          audio.port1.postMessage({ type: 'audio', samples }, [samples.buffer]);
        }
        if (posted >= capture.length) resolve();
        else setTimeout(tick, 10);
      };
      tick();
    });
    audio.port1.postMessage({ type: 'flush' });
    await result;
    return { vad, log, transcript: finals.join(' '), ms: performance.now() - startMs };
  } finally {
    audio.port1.close();
    segments.port2.close();
    worker.terminate();
  }
}
