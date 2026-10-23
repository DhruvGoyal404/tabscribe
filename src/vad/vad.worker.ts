// VAD worker: native-rate capture audio in, speech segments out. It runs apart from the ASR worker
// because a Whisper call blocks its thread for seconds on WASM, and voice detection mustn't stall
// behind it. Audio arrives on one MessagePort (from the capture worklet) and segments leave on
// another (to the ASR worker), so neither passes through a page's main thread.
//
// Silero VAD runs on ONNX Runtime's WASM backend, from the same bundled runtime binary as the ASR
// worker and a model file bundled with the extension. If it can't load, frame energy is used.
import * as ort from 'onnxruntime-web/webgpu';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import { ASR_SAMPLE_RATE, Resampler } from '../asr/resample';
import { EnergyVad } from './energy-vad';
import sileroUrl from './model/silero_vad_16k_op15.onnx?url';
import {
  isAudioPortMessage,
  isVadRequest,
  type VadKind,
  type VadRequest,
  type VadResponse,
  type SegmentPortMessage,
  type VadStats,
} from './protocol';
import { DEFAULT_SEGMENTER, Segmenter, type SegmentEvent } from './segmenter';

ort.env.wasm.wasmPaths = { wasm: new URL(ortWasmUrl, self.location.href).href };
// Not cross-origin isolated, so ORT would fall back to one thread anyway; say so explicitly.
ort.env.wasm.numThreads = 1;

/** Silero's frame at 16 kHz: 512 new samples (32 ms), with 64 samples of context before them. */
const FRAME = 512;
const CONTEXT = 64;

interface FrameVad {
  readonly kind: VadKind;
  probability(frame: Float32Array): Promise<number>;
}

class SileroVad implements FrameVad {
  readonly kind = 'silero';
  private state = new ort.Tensor('float32', new Float32Array(2 * 128), [2, 1, 128]);
  private context = new Float32Array(CONTEXT);
  private readonly sampleRate = new ort.Tensor('int64', BigInt64Array.from([16_000n]), []);

  private constructor(private readonly session: ort.InferenceSession) {}

  static async create(): Promise<SileroVad> {
    const response = await fetch(sileroUrl);
    if (!response.ok) throw new Error(`Silero model: HTTP ${String(response.status)}`);
    const model = new Uint8Array(await response.arrayBuffer());
    return new SileroVad(
      await ort.InferenceSession.create(model, { executionProviders: ['wasm'] }),
    );
  }

  async probability(frame: Float32Array): Promise<number> {
    const input = new Float32Array(CONTEXT + FRAME);
    input.set(this.context);
    input.set(frame, CONTEXT);
    const output = await this.session.run({
      input: new ort.Tensor('float32', input, [1, CONTEXT + FRAME]),
      state: this.state,
      sr: this.sampleRate,
    });
    const next = output.stateN;
    const speech = output.output?.data[0];
    if (!next || typeof speech !== 'number') throw new Error('Unexpected Silero output');
    this.state = next as typeof this.state;
    this.context = input.slice(-CONTEXT);
    return speech;
  }
}

class EnergyFrameVad implements FrameVad {
  readonly kind = 'energy';
  private readonly vad = new EnergyVad({ sampleRate: ASR_SAMPLE_RATE });

  probability(frame: Float32Array): Promise<number> {
    return Promise.resolve(this.vad.probability(frame));
  }
}

interface Session {
  vad: FrameVad;
  resampler: Resampler;
  segmenter: Segmenter;
  audio: MessagePort;
  segments: MessagePort;
  /** 16 kHz samples not yet making up a whole frame. */
  pending: Float32Array;
  stats: {
    frames: number;
    speechFrames: number;
    vadMs: number;
    segments: number;
    partials: number;
  };
}

let session: Session | null = null;
// Audio chunks and control requests are processed strictly in arrival order.
let queue: Promise<void> = Promise.resolve();

const post = (message: VadResponse): void => {
  self.postMessage(message);
};

const emptyStats: Session['stats'] = {
  frames: 0,
  speechFrames: 0,
  vadMs: 0,
  segments: 0,
  partials: 0,
};

function snapshot(stats: Session['stats']): VadStats {
  return {
    frames: stats.frames,
    speechFrames: stats.speechFrames,
    msPerFrame: stats.frames > 0 ? stats.vadMs / stats.frames : 0,
    segments: stats.segments,
    partials: stats.partials,
  };
}

function emit(current: Session, events: SegmentEvent[]): void {
  for (const event of events) {
    if (event.type === 'final') current.stats.segments++;
    else current.stats.partials++;
    current.segments.postMessage(event satisfies SegmentPortMessage, [event.audio.buffer]);
  }
}

async function processFrames(current: Session, samples: Float32Array): Promise<void> {
  const merged = new Float32Array(current.pending.length + samples.length);
  merged.set(current.pending);
  merged.set(samples, current.pending.length);
  let offset = 0;
  for (; offset + FRAME <= merged.length; offset += FRAME) {
    const frame = merged.slice(offset, offset + FRAME);
    const started = performance.now();
    const probability = await current.vad.probability(frame);
    current.stats.vadMs += performance.now() - started;
    current.stats.frames++;
    if (probability >= DEFAULT_SEGMENTER.startThreshold) current.stats.speechFrames++;
    emit(current, current.segmenter.push(frame, probability));
  }
  current.pending = merged.slice(offset);
}

async function init(request: Extract<VadRequest, { type: 'init' }>): Promise<void> {
  session?.audio.close();
  session?.segments.close();
  session = null;
  let vad: FrameVad = new EnergyFrameVad();
  let fallbackReason: string | null = null;
  if (request.vad === 'silero') {
    try {
      vad = await SileroVad.create();
    } catch (error) {
      fallbackReason = String(error).slice(0, 1000);
    }
  }
  const current: Session = {
    vad,
    resampler: new Resampler({ inputRate: request.inputRate }),
    segmenter: new Segmenter({ ...request.tuning, sampleRate: ASR_SAMPLE_RATE }),
    audio: request.audio,
    segments: request.segments,
    pending: new Float32Array(0),
    stats: { frames: 0, speechFrames: 0, vadMs: 0, segments: 0, partials: 0 },
  };
  current.audio.onmessage = (event: MessageEvent<unknown>) => {
    const message = event.data;
    if (!isAudioPortMessage(message)) return;
    enqueue(async () => {
      if (session !== current) return;
      if (message.type === 'flush') await flush(current);
      else await processFrames(current, current.resampler.push(message.samples));
    });
  };
  session = current;
  post({ type: 'ready', id: request.id, vad: vad.kind, fallbackReason });
}

async function flush(current: Session): Promise<void> {
  await processFrames(current, current.resampler.flush());
  if (current.pending.length > 0) {
    // Pad the last partial frame with silence so its audio isn't lost.
    const last = new Float32Array(FRAME);
    last.set(current.pending);
    current.pending = new Float32Array(0);
    await processFrames(current, last);
  }
  emit(current, current.segmenter.flush());
  const end: SegmentPortMessage = { type: 'flushed', stats: snapshot(current.stats) };
  current.segments.postMessage(end);
}

function enqueue(task: () => Promise<void>): void {
  queue = queue.then(task).catch((error: unknown) => {
    console.error('[tabscribe] VAD worker error', error);
  });
}

async function handle(request: VadRequest): Promise<void> {
  switch (request.type) {
    case 'init':
      await init(request);
      return;
    case 'stats':
      post({ type: 'stats', id: request.id, stats: snapshot(session?.stats ?? emptyStats) });
      return;
  }
}

self.addEventListener('message', (event: MessageEvent<unknown>) => {
  const request = event.data;
  if (!isVadRequest(request)) {
    console.warn('[tabscribe] VAD worker ignored an invalid request');
    return;
  }
  enqueue(() =>
    handle(request).catch((error: unknown) => {
      post({ type: 'error', id: request.id, message: String(error).slice(0, 1000) });
    }),
  );
});
