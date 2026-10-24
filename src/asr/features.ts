// Whisper's log-mel features, computed only where there is audio.
//
// Whisper always sees 30 s: the feature extractor zero-pads shorter audio to 480 000 samples and
// runs 3000 STFT frames (n_fft 400, hop 160) plus the mel projection over all of them. A 5 s
// utterance is 5/6 padding, and every padded frame comes out as the same value: zero power hits
// the 1e-10 mel floor, and log10_max_norm clamps it to max(-10, max - 8). Since the padded frames
// never raise the maximum, the frames up to just past the audio, plus one all-zero frame, give
// every value of the full 3000-frame result. The rest is that zero frame's value repeated.
//
// Not quite bit for bit: ONNX Runtime's matmul (the mel projection) rounds a few values differently
// with fewer columns, by a float32 step or so.

import { Tensor } from '@huggingface/transformers';

/** A Transformers.js log-mel tensor: [mel bins, frames], row-major. */
export interface MelFeatures {
  data: Float32Array;
  dims: readonly number[];
}

export interface FeatureGeometry {
  /** STFT hop in samples (160). */
  hop: number;
  /** STFT frame length in samples (400); frames are centred, so each reaches half of it back. */
  frameLength: number;
  /** Frames in the full window (3000). */
  maxFrames: number;
}

/** Length of `waveform` without its trailing zeros. */
export function audioLength(waveform: Float32Array): number {
  let end = waveform.length;
  while (end > 0 && waveform[end - 1] === 0) end -= 1;
  return end;
}

/**
 * Samples to run the extractor on so that its last frame is all zeros and every earlier frame
 * matches the full window's, or null when the audio fills (nearly) the whole window.
 */
export function trimmedLength(
  audioSamples: number,
  totalSamples: number,
  { hop, frameLength, maxFrames }: FeatureGeometry,
): number | null {
  const half = frameLength / 2;
  // The first frame whose centred window starts at or after the audio's end.
  const zeroFrame = Math.ceil((audioSamples + half) / hop);
  if (zeroFrame + 1 >= maxFrames) return null;
  // Ends exactly where frame `zeroFrame` does, so that frame is the extractor's last, and nothing
  // the centring reflects back in is audio.
  const samples = zeroFrame * hop + half;
  return samples < totalSamples ? samples : null;
}

/** Widens [mels, frames] to [mels, maxFrames] by repeating each row's last (all-zero) frame. */
export function padFrames(features: MelFeatures, maxFrames: number): MelFeatures {
  const [mels = 0, frames = 0] = features.dims;
  const data = new Float32Array(mels * maxFrames);
  for (let mel = 0; mel < mels; mel++) {
    const row = features.data.subarray(mel * frames, (mel + 1) * frames);
    const out = data.subarray(mel * maxFrames, (mel + 1) * maxFrames);
    out.set(row);
    out.fill(row[frames - 1] ?? 0, frames);
  }
  return { data, dims: [mels, maxFrames] };
}

/** The parts of Transformers.js's WhisperFeatureExtractor this relies on. */
export interface WhisperFeatureExtractorLike {
  config: { hop_length: number; n_fft: number; nb_max_frames: number };
  _extract_fbank_features(waveform: Float32Array): Promise<Tensor>;
}

/**
 * Makes `extractor` compute only the frames that see audio. Its output is unchanged up to float32
 * rounding (features.test.ts checks this against the unpatched extractor).
 */
export function skipPaddingFrames(extractor: WhisperFeatureExtractorLike): void {
  const full = extractor._extract_fbank_features.bind(extractor);
  const geometry: FeatureGeometry = {
    hop: extractor.config.hop_length,
    frameLength: extractor.config.n_fft,
    maxFrames: extractor.config.nb_max_frames,
  };
  extractor._extract_fbank_features = async (waveform) => {
    const samples = trimmedLength(audioLength(waveform), waveform.length, geometry);
    if (samples === null) return full(waveform);
    const trimmed = await full(waveform.subarray(0, samples));
    const { data, dims } = padFrames(
      { data: trimmed.data as Float32Array, dims: trimmed.dims },
      geometry.maxFrames,
    );
    return new Tensor('float32', data, [...dims]);
  };
}
