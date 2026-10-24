// Runs on the audio rendering thread. Bundled as a standalone module (see the `?worker&url`
// import in the offscreen document) and loaded with audioWorklet.addModule. Two silent taps:
// - the level meter, posting ~20 levels a second to the offscreen document;
// - the capture tap, posting mono audio in 2048-sample chunks straight to the VAD worker, over a
//   MessagePort the offscreen document hands it, so audio never waits on a page's main thread.
import { CAPTURE_PROCESSOR, Chunker } from '../shared/chunker';
import { blockSizeFor, downmix, LEVEL_METER_PROCESSOR, LevelMeter } from '../shared/level';

/** Mono view of the first input, reusing `buffer` when the quantum size matches. */
function monoInput(inputs: Float32Array[][], buffer: Float32Array): Float32Array | null {
  const channels = inputs[0];
  const frames = channels?.[0]?.length ?? 0;
  if (!channels || frames === 0) return null;
  return downmix(channels, buffer.length === frames ? buffer : new Float32Array(frames));
}

const isPort = (value: unknown): value is WorkletPort =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as Partial<WorkletPort>).postMessage === 'function';

class LevelMeterProcessor extends AudioWorkletProcessor {
  private readonly meter = new LevelMeter(blockSizeFor(sampleRate));
  private mono: Float32Array = new Float32Array(128);

  override process(inputs: Float32Array[][]): boolean {
    const mono = monoInput(inputs, this.mono);
    if (mono) {
      this.mono = mono;
      // One level per ~50 ms block, which throttles posts to ~20 Hz.
      const level = this.meter.push(mono);
      if (level) this.port.postMessage(level);
    }
    // Outputs stay zero-filled: this node is a silent tap.
    return true;
  }
}

class CaptureTapProcessor extends AudioWorkletProcessor {
  private readonly chunker = new Chunker();
  private mono: Float32Array = new Float32Array(128);
  private out: WorkletPort | null = null;

  constructor() {
    super();
    // The offscreen document sends the VAD worker's audio port once, right after creating the node.
    this.port.onmessage = (event) => {
      if (isPort(event.data)) this.out = event.data;
    };
  }

  override process(inputs: Float32Array[][]): boolean {
    const mono = monoInput(inputs, this.mono);
    if (mono && this.out) {
      this.mono = mono;
      for (const samples of this.chunker.push(mono)) {
        this.out.postMessage({ type: 'audio', samples }, [samples.buffer]);
      }
    }
    return true;
  }
}

registerProcessor(LEVEL_METER_PROCESSOR, LevelMeterProcessor);
registerProcessor(CAPTURE_PROCESSOR, CaptureTapProcessor);
