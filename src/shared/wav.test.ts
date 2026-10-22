import { describe, expect, it } from 'vitest';
import { encodeWav, floatTo16BitPcm } from './wav';

const ascii = (bytes: Uint8Array, start: number, length: number) =>
  String.fromCharCode(...bytes.subarray(start, start + length));

describe('floatTo16BitPcm', () => {
  it('maps the full range asymmetrically to int16 bounds', () => {
    expect(Array.from(floatTo16BitPcm(new Float32Array([-1, 0, 1])))).toEqual([-32768, 0, 32767]);
  });

  it('scales intermediate values and truncates toward zero', () => {
    expect(Array.from(floatTo16BitPcm(new Float32Array([0.5, -0.5])))).toEqual([16383, -16384]);
  });

  it('clamps out-of-range samples', () => {
    expect(Array.from(floatTo16BitPcm(new Float32Array([2, -3, Infinity, -Infinity])))).toEqual([
      32767, -32768, 32767, -32768,
    ]);
  });

  it('turns NaN into silence', () => {
    expect(Array.from(floatTo16BitPcm(new Float32Array([NaN])))).toEqual([0]);
  });

  it('returns an empty array for empty input', () => {
    expect(floatTo16BitPcm(new Float32Array(0))).toHaveLength(0);
  });
});

describe('encodeWav', () => {
  it('writes a canonical mono 16 kHz header', () => {
    const wav = encodeWav(new Int16Array(16000), 16000);
    const view = new DataView(wav.buffer);

    expect(wav.byteLength).toBe(44 + 32000);
    expect(ascii(wav, 0, 4)).toBe('RIFF');
    expect(view.getUint32(4, true)).toBe(36 + 32000);
    expect(ascii(wav, 8, 4)).toBe('WAVE');
    expect(ascii(wav, 12, 4)).toBe('fmt ');
    expect(view.getUint32(16, true)).toBe(16);
    expect(view.getUint16(20, true)).toBe(1);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint32(28, true)).toBe(32000);
    expect(view.getUint16(32, true)).toBe(2);
    expect(view.getUint16(34, true)).toBe(16);
    expect(ascii(wav, 36, 4)).toBe('data');
    expect(view.getUint32(40, true)).toBe(32000);
  });

  it('writes samples little-endian after the header', () => {
    const wav = encodeWav(new Int16Array([1, -2, 32767, -32768]), 8000);
    const view = new DataView(wav.buffer);
    expect([0, 1, 2, 3].map((i) => view.getInt16(44 + i * 2, true))).toEqual([
      1, -2, 32767, -32768,
    ]);
    expect(Array.from(wav.subarray(44, 46))).toEqual([0x01, 0x00]);
  });

  it('computes byte rate and block align for stereo', () => {
    const view = new DataView(encodeWav(new Int16Array(4), 44100, 2).buffer);
    expect(view.getUint16(22, true)).toBe(2);
    expect(view.getUint32(28, true)).toBe(44100 * 4);
    expect(view.getUint16(32, true)).toBe(4);
  });

  it('produces a header-only file for empty input', () => {
    const wav = encodeWav(new Int16Array(0), 16000);
    expect(wav.byteLength).toBe(44);
    expect(new DataView(wav.buffer).getUint32(40, true)).toBe(0);
  });

  it.each([0, -1, 1.5, NaN])('rejects invalid sample rate %s', (rate) => {
    expect(() => encodeWav(new Int16Array(2), rate)).toThrow(RangeError);
  });

  it('rejects invalid channel counts', () => {
    expect(() => encodeWav(new Int16Array(2), 16000, 0)).toThrow(RangeError);
    expect(() => encodeWav(new Int16Array(3), 16000, 2)).toThrow(RangeError);
  });

  it('round-trips float audio through PCM', () => {
    const wav = encodeWav(floatTo16BitPcm(new Float32Array([0.25, -0.25])), 16000);
    const view = new DataView(wav.buffer);
    expect(view.getInt16(44, true)).toBe(8191);
    expect(view.getInt16(46, true)).toBe(-8192);
  });
});
