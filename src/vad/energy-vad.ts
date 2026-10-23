// Fallback VAD for when the Silero model can't be loaded: frame energy against an adaptive noise
// floor. It can't tell speech from music or noise bursts, so the ASR side's no-speech check matters
// more when this is in use; the debug view says which VAD is active.
import { toDbfs } from '../shared/level';

export interface EnergyVadOptions {
  sampleRate: number;
  /** Speech must be this far above the noise floor. */
  marginDb: number;
  /** Anything quieter is never speech, however quiet the floor. */
  absoluteFloorDbfs: number;
  /** How fast the floor creeps up under steady sound, so constant noise stops counting as speech. */
  riseDbPerSecond: number;
}

export const DEFAULT_ENERGY_VAD: EnergyVadOptions = {
  sampleRate: 16_000,
  marginDb: 10,
  absoluteFloorDbfs: -55,
  riseDbPerSecond: 1.5,
};

/** dB either side of the threshold over which the output ramps from 0 to 1. */
const RAMP_DB = 6;

export class EnergyVad {
  private readonly options: EnergyVadOptions;
  private floorDb: number | null = null;

  constructor(options: Partial<EnergyVadOptions> = {}) {
    this.options = { ...DEFAULT_ENERGY_VAD, ...options };
  }

  /** A speech probability in [0, 1] for one frame, comparable with Silero's thresholds. */
  probability(frame: Float32Array): number {
    let sum = 0;
    for (const sample of frame) sum += sample * sample;
    const db = toDbfs(Math.sqrt(sum / Math.max(1, frame.length)));
    const seconds = frame.length / this.options.sampleRate;
    // The floor drops at once to quieter frames and rises slowly under louder ones.
    this.floorDb =
      this.floorDb === null || db < this.floorDb
        ? db
        : this.floorDb + this.options.riseDbPerSecond * seconds;
    const threshold = Math.max(
      this.floorDb + this.options.marginDb,
      this.options.absoluteFloorDbfs,
    );
    return Math.min(1, Math.max(0, 0.5 + (db - threshold) / (2 * RAMP_DB)));
  }

  reset(): void {
    this.floorDb = null;
  }
}
