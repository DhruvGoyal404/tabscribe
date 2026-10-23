// Streams native-rate audio through a fresh VAD worker, the way the capture worklet will: chunks
// on one MessagePort, segments back on another, end-of-stream on the data path.
import {
  isSegmentPortMessage,
  isVadResponse,
  type SegmenterTuning,
  type VadKind,
  type VadStats,
} from '../vad/protocol';
import type { SegmentEvent } from '../vad/segmenter';

/** What a 128-frame worklet would post after batching 16 render quanta (about 43 ms at 48 kHz). */
const CHUNK = 2048;

export interface VadRun {
  vad: VadKind;
  fallbackReason: string | null;
  events: SegmentEvent[];
  stats: VadStats;
  /** Wall-clock time for the whole stream, as fast as the worker could go. */
  ms: number;
}

export async function runVad(
  audio: Float32Array,
  inputRate: number,
  vad: VadKind,
  tuning: SegmenterTuning = {},
): Promise<VadRun> {
  const worker = new Worker(new URL('../vad/vad.worker.ts', import.meta.url), { type: 'module' });
  const audioChannel = new MessageChannel();
  const segmentChannel = new MessageChannel();
  try {
    const ready = await new Promise<{ vad: VadKind; fallbackReason: string | null }>(
      (resolve, reject) => {
        worker.onmessage = (event: MessageEvent<unknown>) => {
          const response = event.data;
          if (!isVadResponse(response)) return;
          if (response.type === 'ready') resolve(response);
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
            vad,
            tuning,
            audio: audioChannel.port2,
            segments: segmentChannel.port1,
          },
          [audioChannel.port2, segmentChannel.port1],
        );
      },
    );

    const events: SegmentEvent[] = [];
    const started = performance.now();
    const done = new Promise<VadStats>((resolve) => {
      segmentChannel.port2.onmessage = (event: MessageEvent<unknown>) => {
        const message = event.data;
        if (!isSegmentPortMessage(message)) return;
        if (message.type === 'flushed') resolve(message.stats);
        else events.push(message);
      };
    });
    for (let offset = 0; offset < audio.length; offset += CHUNK) {
      const samples = audio.slice(offset, offset + CHUNK);
      audioChannel.port1.postMessage({ type: 'audio', samples }, [samples.buffer]);
    }
    audioChannel.port1.postMessage({ type: 'flush' });
    const stats = await done;
    return { ...ready, events, stats, ms: performance.now() - started };
  } finally {
    audioChannel.port1.close();
    segmentChannel.port2.close();
    worker.terminate();
  }
}
