// Band-limited resampling from the capture's native rate (usually 48 or 44.1 kHz) to the 16 kHz
// mono that Whisper expects. Decimating without a low-pass would fold everything above 8 kHz
// (sibilants, cymbals, music) back into the speech band as aliasing.
//
// The filter is a Kaiser-windowed sinc evaluated at the exact fractional position of each output
// sample, so any rate pair works. Positions are tracked with integer arithmetic (no drift over
// long captures), and the per-phase kernels are precomputed whenever the rate pair has few phases
// (48k -> 16k has one, 44.1k -> 16k has 160).

export const ASR_SAMPLE_RATE = 16_000;

/** Passband edge as a fraction of the lower Nyquist: -6 dB at 90%, stopband from 100%. */
const CUTOFF = 0.9;
/** Stopband attenuation target in dB. */
const ATTENUATION_DB = 80;
/** Above this many distinct phases, kernels are computed per sample instead of tabulated. */
const MAX_TABLE_PHASES = 4096;

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));

/** Zeroth-order modified Bessel function of the first kind (power series). */
function besselI0(x: number): number {
  let sum = 1;
  let term = 1;
  const quarterX2 = (x * x) / 4;
  for (let k = 1; term > sum * 1e-12; k++) {
    term *= quarterX2 / (k * k);
    sum += term;
  }
  return sum;
}

/** Kaiser beta for a given stopband attenuation (Kaiser's empirical formula). */
function kaiserBeta(attenuationDb: number): number {
  if (attenuationDb > 50) return 0.1102 * (attenuationDb - 8.7);
  if (attenuationDb >= 21) {
    return 0.5842 * (attenuationDb - 21) ** 0.4 + 0.07886 * (attenuationDb - 21);
  }
  return 0;
}

export interface ResamplerOptions {
  inputRate: number;
  outputRate?: number;
}

/**
 * Streaming resampler: push native-rate mono chunks of any size, get output-rate samples back.
 * Output is identical however the input is split into chunks. Call flush() at the end.
 */
export class Resampler {
  readonly inputRate: number;
  readonly outputRate: number;
  /** Kernel half-width in input samples. */
  private readonly halfWidth: number;
  /** Cutoff in cycles per input sample. */
  private readonly cutoff: number;
  private readonly beta: number;
  private readonly i0Beta: number;
  /** Distinct fractional offsets: outputRate / gcd. */
  private readonly phases: number;
  private readonly table: Float32Array[] | null;

  /** Pending input; buffer[0] is absolute input index `bufferStart`. */
  private buffer = new Float32Array(0);
  private bufferStart = 0;
  private inputCount = 0;
  /** Absolute index of the next output sample. */
  private nextOutput = 0;

  constructor({ inputRate, outputRate = ASR_SAMPLE_RATE }: ResamplerOptions) {
    if (!Number.isInteger(inputRate) || inputRate <= 0) throw new RangeError('Bad input rate');
    if (!Number.isInteger(outputRate) || outputRate <= 0) throw new RangeError('Bad output rate');
    this.inputRate = inputRate;
    this.outputRate = outputRate;

    const nyquist = Math.min(inputRate, outputRate) / 2;
    this.cutoff = (CUTOFF * nyquist) / inputRate;
    // Transition band: from the cutoff to the lower Nyquist, on both sides of -6 dB.
    const transition = (2 * (1 - CUTOFF) * nyquist) / inputRate;
    const taps = (ATTENUATION_DB - 7.95) / (2.285 * 2 * Math.PI * transition);
    this.halfWidth = Math.ceil(taps / 2);
    this.beta = kaiserBeta(ATTENUATION_DB);
    this.i0Beta = besselI0(this.beta);

    this.phases = outputRate / gcd(inputRate, outputRate);
    this.table =
      inputRate === outputRate || this.phases > MAX_TABLE_PHASES
        ? null
        : Array.from({ length: this.phases }, (_, phase) => this.kernel(phase / this.phases));
  }

  /** Filter taps for input indices base-halfWidth+1 .. base+halfWidth, normalised to unity DC. */
  private kernel(fraction: number): Float32Array {
    const taps = new Float32Array(2 * this.halfWidth);
    let sum = 0;
    for (let i = 0; i < taps.length; i++) {
      // Distance from the output position to input sample (base - halfWidth + 1 + i).
      const distance = i - this.halfWidth + 1 - fraction;
      const ratio = distance / this.halfWidth;
      if (Math.abs(ratio) >= 1) continue;
      const window = besselI0(this.beta * Math.sqrt(1 - ratio * ratio)) / this.i0Beta;
      const x = 2 * this.cutoff * distance;
      const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      taps[i] = 2 * this.cutoff * sinc * window;
      sum += taps[i] ?? 0;
    }
    for (let i = 0; i < taps.length; i++) taps[i] = (taps[i] ?? 0) / sum;
    return taps;
  }

  /** Total output samples for `inputCount` input samples: positions strictly inside the input. */
  private outputLength(inputCount: number): number {
    return Math.ceil((inputCount * this.outputRate) / this.inputRate);
  }

  push(input: Float32Array): Float32Array {
    if (this.inputRate === this.outputRate) {
      this.inputCount += input.length;
      this.nextOutput = this.inputCount;
      return input.slice();
    }
    const merged = new Float32Array(this.buffer.length + input.length);
    merged.set(this.buffer);
    merged.set(input, this.buffer.length);
    this.buffer = merged;
    this.inputCount += input.length;
    return this.drain(this.inputCount);
  }

  /** Emits the remaining output, treating input past the end as silence, and resets the stream. */
  flush(): Float32Array {
    if (this.inputRate === this.outputRate) {
      this.reset();
      return new Float32Array(0);
    }
    const padded = new Float32Array(this.buffer.length + this.halfWidth);
    padded.set(this.buffer);
    this.buffer = padded;
    const out = this.drain(this.inputCount + this.halfWidth, this.outputLength(this.inputCount));
    this.reset();
    return out;
  }

  reset(): void {
    this.buffer = new Float32Array(0);
    this.bufferStart = 0;
    this.inputCount = 0;
    this.nextOutput = 0;
  }

  /** Computes outputs whose kernel lies within the first `available` input samples. */
  private drain(available: number, limit = Infinity): Float32Array {
    const out: number[] = [];
    for (;;) {
      if (this.nextOutput >= limit) break;
      // Exact position: nextOutput * inputRate / outputRate = base + remainder / outputRate.
      const numerator = this.nextOutput * this.inputRate;
      const base = Math.floor(numerator / this.outputRate);
      if (base + this.halfWidth >= available) break;
      const remainder = numerator - base * this.outputRate;
      out.push(this.sample(base, remainder));
      this.nextOutput++;
    }
    // Keep only the history the next output's kernel can still reach.
    const nextBase = Math.floor((this.nextOutput * this.inputRate) / this.outputRate);
    const keepFrom = Math.max(this.bufferStart, nextBase - this.halfWidth + 1);
    if (keepFrom > this.bufferStart) {
      this.buffer = this.buffer.slice(keepFrom - this.bufferStart);
      this.bufferStart = keepFrom;
    }
    return Float32Array.from(out);
  }

  private sample(base: number, remainder: number): number {
    const phase = (remainder * this.phases) / this.outputRate;
    const taps = this.table?.[phase] ?? this.kernel(remainder / this.outputRate);
    const first = base - this.halfWidth + 1;
    let acc = 0;
    for (let i = 0; i < taps.length; i++) {
      const index = first + i - this.bufferStart;
      // Before the stream start (index < 0) counts as silence.
      if (index >= 0 && index < this.buffer.length)
        acc += (taps[i] ?? 0) * (this.buffer[index] ?? 0);
    }
    return acc;
  }
}

/** Averages channels into one; a single channel is returned as is. */
export function downmixToMono(channels: readonly Float32Array[]): Float32Array {
  const [first, ...rest] = channels;
  if (!first) return new Float32Array(0);
  if (rest.length === 0) return first;
  const out = new Float32Array(first.length);
  for (const channel of channels) {
    if (channel.length !== first.length) throw new RangeError('Channel lengths differ');
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) + (channel[i] ?? 0);
  }
  const scale = 1 / channels.length;
  for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) * scale;
  return out;
}

/** One-shot helper: downmix, then resample a whole buffer to 16 kHz. */
export function resampleTo16k(channels: readonly Float32Array[], inputRate: number): Float32Array {
  const resampler = new Resampler({ inputRate });
  const head = resampler.push(downmixToMono(channels));
  const tail = resampler.flush();
  const out = new Float32Array(head.length + tail.length);
  out.set(head);
  out.set(tail, head.length);
  return out;
}
