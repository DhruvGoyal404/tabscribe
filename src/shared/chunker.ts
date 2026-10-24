// Batches the audio thread's 128-frame render quanta into larger chunks for the VAD worker, so the
// capture tap posts about 23 messages a second at 48 kHz instead of 375. Free of DOM APIs: it is
// bundled into the worklet.

export const CAPTURE_PROCESSOR = 'capture-tap';
/** 16 render quanta: about 43 ms at 48 kHz, as the bench's VAD runs use. */
export const CAPTURE_CHUNK = 2048;

export class Chunker {
  private chunk: Float32Array;
  private filled = 0;

  constructor(private readonly size = CAPTURE_CHUNK) {
    this.chunk = new Float32Array(size);
  }

  /** Copies `samples` in and returns every chunk that filled up (each a fresh buffer). */
  push(samples: Float32Array): Float32Array[] {
    const full: Float32Array[] = [];
    let offset = 0;
    while (offset < samples.length) {
      const count = Math.min(this.size - this.filled, samples.length - offset);
      this.chunk.set(samples.subarray(offset, offset + count), this.filled);
      this.filled += count;
      offset += count;
      if (this.filled === this.size) {
        full.push(this.chunk);
        this.chunk = new Float32Array(this.size);
        this.filled = 0;
      }
    }
    return full;
  }
}
