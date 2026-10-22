const WAV_HEADER_BYTES = 44;
const BYTES_PER_SAMPLE = 2;

/**
 * Converts float samples in [-1, 1] to signed 16-bit PCM.
 * Out-of-range values are clamped; NaN becomes silence.
 */
export function floatTo16BitPcm(samples: Float32Array): Int16Array {
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i] ?? 0;
    const clamped = Number.isNaN(sample) ? 0 : Math.max(-1, Math.min(1, sample));
    pcm[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return pcm;
}

/**
 * Wraps 16-bit PCM (interleaved when channels > 1) in a canonical 44-byte RIFF/WAVE header.
 * Samples are written little-endian regardless of platform byte order.
 */
export function encodeWav(pcm: Int16Array, sampleRate: number, channels = 1): Uint8Array {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`sampleRate must be a positive integer, got ${String(sampleRate)}`);
  }
  if (!Number.isInteger(channels) || channels < 1) {
    throw new RangeError(`channels must be a positive integer, got ${String(channels)}`);
  }
  if (pcm.length % channels !== 0) {
    throw new RangeError(
      `sample count ${String(pcm.length)} is not divisible by ${String(channels)} channels`,
    );
  }

  const dataBytes = pcm.length * BYTES_PER_SAMPLE;
  const blockAlign = channels * BYTES_PER_SAMPLE;
  const buffer = new ArrayBuffer(WAV_HEADER_BYTES + dataBytes);
  const view = new DataView(buffer);

  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true); // byte rate
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, BYTES_PER_SAMPLE * 8, true); // bits per sample
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataBytes, true);

  for (let i = 0; i < pcm.length; i++) {
    view.setInt16(WAV_HEADER_BYTES + i * BYTES_PER_SAMPLE, pcm[i] ?? 0, true);
  }
  return new Uint8Array(buffer);
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}
