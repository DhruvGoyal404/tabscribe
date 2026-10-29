// The caption pipeline in the offscreen document:
//
//   capture tap (audio thread) ──port──▶ VAD worker ──port──▶ this page: CaptionScheduler ──▶ ASR worker
//
// The model loads first, with progress; only then does audio flow to the VAD, so a long download
// doesn't pile up segments. Segments pass through this page, which does little else. Stopping
// terminates both workers, which frees the model.
import { CAPTION_MODEL, CAPTION_PROMPTING, planBytes, type CaptionPlan } from '../asr/backend';
import { isWorkerResponse, type WorkerRequest, type WorkerResponse } from '../asr/protocol';
import { CaptionScheduler, type DecodeJob } from '../asr/scheduler';
import type { Caption, CaptionStatus } from '../shared/captions';
import { CAPTURE_PROCESSOR } from '../shared/chunker';
import { describeError } from '../shared/messages';
import { DownloadRate } from './download-rate';
import { isSegmentPortMessage, isVadResponse, type VadKind } from '../vad/protocol';

export interface CaptionEvents {
  status(status: CaptionStatus): void;
  caption(caption: Caption): void;
  /** The planned WebGPU backend failed to load. */
  webgpuFailed(message: string): void;
}

/** Request id for the model load; decode jobs count up from 0. */
const LOAD_ID = Number.MAX_SAFE_INTEGER;

export class CaptionPipeline {
  private readonly asr: Worker;
  private vad: Worker | null = null;
  private tap: AudioWorkletNode | null = null;
  private segments: MessageChannel | null = null;
  private scheduler: CaptionScheduler | null = null;
  private stopped = false;

  private constructor(
    private readonly plan: CaptionPlan,
    private readonly events: CaptionEvents,
  ) {
    this.asr = new Worker(new URL('../asr/asr.worker.ts', import.meta.url), { type: 'module' });
  }

  /**
   * Loads the model, then taps `source`. Never rejects: failures end the captions with an error
   * status, while capture and playback go on.
   */
  static start(
    context: AudioContext,
    source: AudioNode,
    plan: CaptionPlan,
    allowDownload: boolean,
    events: CaptionEvents,
  ): CaptionPipeline {
    const pipeline = new CaptionPipeline(plan, events);
    void pipeline.run(context, source, allowDownload);
    return pipeline;
  }

  stop(): void {
    if (this.isStopped()) return;
    this.stopped = true;
    this.tap?.disconnect();
    this.segments?.port2.close();
    this.vad?.terminate();
    this.asr.terminate();
  }

  /** A method, not the field, so TypeScript doesn't keep it narrowed across awaits. */
  private isStopped(): boolean {
    return this.stopped;
  }

  private async run(context: AudioContext, source: AudioNode, allowDownload: boolean) {
    try {
      await this.load(allowDownload);
      if (this.isStopped()) return;
      const vad = await this.listen(context, source);
      if (this.isStopped()) return;
      this.events.status({ state: 'listening', plan: this.plan, vad });
    } catch (error) {
      if (this.isStopped()) return;
      const message = describeError(error);
      if (this.plan.device === 'webgpu' && !message.includes('ModelNotDownloaded')) {
        this.events.webgpuFailed(message);
      }
      this.events.status({ state: 'error', plan: this.plan, message });
      this.stop();
    }
  }

  private load(allowDownload: boolean): Promise<void> {
    const totalBytes = allowDownload ? planBytes(this.plan) : 0;
    const files = new Map<string, number>();
    const rate = new DownloadRate();
    this.events.status({
      state: 'loading',
      plan: this.plan,
      downloading: allowDownload,
      loadedBytes: 0,
      totalBytes,
      bytesPerSecond: null,
    });
    return new Promise((resolve, reject) => {
      this.asr.onerror = (event) => {
        reject(new Error(event.message || 'The ASR worker failed to start.'));
      };
      this.asr.onmessage = (event: MessageEvent<unknown>) => {
        const response = event.data;
        if (!isWorkerResponse(response)) return;
        if (response.type === 'progress') {
          files.set(response.file, response.loaded);
          const loadedBytes = [...files.values()].reduce((sum, bytes) => sum + bytes, 0);
          this.events.status({
            state: 'loading',
            plan: this.plan,
            downloading: allowDownload,
            loadedBytes: Math.min(loadedBytes, totalBytes),
            totalBytes,
            bytesPerSecond: rate.update(performance.now(), loadedBytes),
          });
        } else if (response.type === 'loaded') resolve();
        else if (response.type === 'error' && response.id === LOAD_ID) {
          reject(new Error(response.message));
        } else this.onAsr(response);
      };
      this.post({
        type: 'load',
        id: LOAD_ID,
        model: CAPTION_MODEL,
        device: this.plan.device,
        dtype: this.plan.dtype,
        threads: this.plan.threads,
        allowDownload,
      });
    });
  }

  /** Starts the VAD and connects the capture tap to it; resolves with the VAD in use. */
  private async listen(context: AudioContext, source: AudioNode): Promise<VadKind> {
    const { device } = this.plan;
    this.scheduler = new CaptionScheduler(
      {
        partials: device === 'webgpu',
        mergeQueued: device === 'wasm',
        prompting: CAPTION_PROMPTING,
      },
      {
        decode: (job) => {
          this.decode(job);
        },
        interrupt: (id) => {
          this.post({ type: 'interrupt', id });
        },
      },
      (caption) => {
        if (caption.type === 'partial') {
          this.events.caption({
            kind: 'partial',
            segmentId: caption.segmentId,
            text: caption.text,
          });
        } else {
          const segmentId = caption.segmentIds.at(-1) ?? 0;
          this.events.caption({ kind: 'final', segmentId, text: caption.text });
        }
      },
    );

    const audio = new MessageChannel();
    const segments = new MessageChannel();
    this.segments = segments;
    const vad = new Worker(new URL('../vad/vad.worker.ts', import.meta.url), { type: 'module' });
    this.vad = vad;
    const ready = await new Promise<VadKind>((resolve, reject) => {
      vad.onerror = (event) => {
        reject(new Error(event.message || 'The VAD worker failed to start.'));
      };
      vad.onmessage = (event: MessageEvent<unknown>) => {
        const response = event.data;
        if (!isVadResponse(response)) return;
        if (response.type === 'ready') resolve(response.vad);
        else if (response.type === 'error') reject(new Error(response.message));
      };
      vad.postMessage(
        {
          type: 'init',
          id: 0,
          inputRate: context.sampleRate,
          vad: 'silero',
          tuning: {},
          audio: audio.port1,
          segments: segments.port1,
        },
        [audio.port1, segments.port1],
      );
    });
    segments.port2.onmessage = (event: MessageEvent<unknown>) => {
      const message = event.data;
      if (isSegmentPortMessage(message) && message.type !== 'flushed') {
        this.scheduler?.push(message);
      }
    };
    const tap = new AudioWorkletNode(context, CAPTURE_PROCESSOR);
    this.tap = tap;
    tap.port.postMessage(audio.port2, [audio.port2]);
    source.connect(tap);
    // The tap's output is silent; connecting it keeps the node in the rendered graph.
    tap.connect(context.destination);
    return ready;
  }

  private decode(job: DecodeJob): void {
    this.post(
      { type: 'decode', id: job.id, audio: job.audio, kind: job.kind, prompt: job.prompt },
      [job.audio.buffer],
    );
  }

  private onAsr(response: WorkerResponse): void {
    if (response.type === 'decoded') this.scheduler?.onDecoded(response);
    else if (response.type === 'error') {
      this.events.status({ state: 'error', plan: this.plan, message: response.message });
      this.stop();
    }
  }

  private post(request: WorkerRequest, transfer: Transferable[] = []): void {
    this.asr.postMessage(request, transfer);
  }
}
