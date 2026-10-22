// Level metering shared by the AudioWorklet and its tests. Must stay free of DOM APIs:
// it is bundled into the worklet, which runs in an AudioWorkletGlobalScope.

/** Floor for silence, so the meter never has to render -Infinity. */
export const MIN_DBFS = -100;
export const LEVEL_METER_PROCESSOR = 'level-meter';
export const LEVEL_BLOCK_MS = 50;

export interface Level {
  /** RMS level in dBFS (a full-scale sine reads about -3.01). */
  rmsDbfs: number;
  /** Peak sample level in dBFS. */
  peakDbfs: number;
  /** True when any sample in the block went beyond full scale. */
  clipped: boolean;
}

export function toDbfs(amplitude: number): number {
  return amplitude > 0 ? Math.max(MIN_DBFS, 20 * Math.log10(amplitude)) : MIN_DBFS;
}

export function blockSizeFor(sampleRate: number, blockMs = LEVEL_BLOCK_MS): number {
  return Math.max(1, Math.round((sampleRate * blockMs) / 1000));
}

/** Averages channels into `out` (sized to the first channel) and returns it. */
export function downmix(channels: readonly Float32Array[], out: Float32Array): Float32Array {
  out.fill(0);
  if (channels.length === 0) return out;
  for (const channel of channels) {
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) + (channel[i] ?? 0);
  }
  if (channels.length > 1) {
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) / channels.length;
  }
  return out;
}

/** Accumulates mono samples and reports one Level per `blockSize` samples. */
export class LevelMeter {
  private sumSquares = 0;
  private peak = 0;
  private count = 0;

  constructor(private readonly blockSize: number) {
    if (!Number.isInteger(blockSize) || blockSize < 1) {
      throw new RangeError(`blockSize must be a positive integer, got ${String(blockSize)}`);
    }
  }

  /**
   * Adds samples and returns the level of the most recently completed block, or null if no
   * block completed. Non-finite samples count as silence.
   */
  push(samples: Float32Array): Level | null {
    let completed: Level | null = null;
    for (const raw of samples) {
      const sample = Number.isFinite(raw) ? raw : 0;
      this.sumSquares += sample * sample;
      this.peak = Math.max(this.peak, Math.abs(sample));
      if (++this.count === this.blockSize) completed = this.flush();
    }
    return completed;
  }

  private flush(): Level {
    const level: Level = {
      rmsDbfs: toDbfs(Math.sqrt(this.sumSquares / this.count)),
      peakDbfs: toDbfs(this.peak),
      clipped: this.peak > 1,
    };
    this.sumSquares = 0;
    this.peak = 0;
    this.count = 0;
    return level;
  }
}
