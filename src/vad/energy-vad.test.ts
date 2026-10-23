import { describe, expect, it } from 'vitest';
import { EnergyVad } from './energy-vad';

const RATE = 16_000;
const FRAME = 512;

/** A frame of a 300 Hz tone at the given peak level (dBFS). */
function tone(dbfs: number, index = 0): Float32Array {
  const amplitude = 10 ** (dbfs / 20);
  return Float32Array.from(
    { length: FRAME },
    (_, i) => amplitude * Math.sin((2 * Math.PI * 300 * (index * FRAME + i)) / RATE),
  );
}

const framesFor = (seconds: number) => Math.round((seconds * RATE) / FRAME);

describe('EnergyVad', () => {
  it('reports silence as non-speech', () => {
    const vad = new EnergyVad();
    for (let i = 0; i < 20; i++) expect(vad.probability(new Float32Array(FRAME))).toBe(0);
  });

  it('flags a loud sound over a quiet floor as speech', () => {
    const vad = new EnergyVad();
    for (let i = 0; i < framesFor(1); i++) vad.probability(tone(-60, i));
    expect(vad.probability(tone(-20))).toBe(1);
  });

  it('never flags very quiet sound, even over digital silence', () => {
    const vad = new EnergyVad();
    for (let i = 0; i < 10; i++) vad.probability(new Float32Array(FRAME));
    expect(vad.probability(tone(-70))).toBe(0);
  });

  it('adapts to steady noise, so constant hum stops counting as speech', () => {
    const vad = new EnergyVad();
    for (let i = 0; i < framesFor(1); i++) vad.probability(tone(-60, i));
    expect(vad.probability(tone(-30))).toBe(1);
    let last = 1;
    for (let i = 0; i < framesFor(30); i++) last = vad.probability(tone(-30, i));
    expect(last).toBeLessThan(0.35);
  });

  it('ramps smoothly around the threshold', () => {
    const vad = new EnergyVad({ absoluteFloorDbfs: -100 });
    for (let i = 0; i < framesFor(1); i++) vad.probability(tone(-60, i));
    // RMS of a sine is its peak - 3 dB: the floor is -63 dB, the threshold -53 dB RMS, which is a
    // -50 dBFS peak. 6 dB either side ramps to 0 and 1.
    expect(vad.probability(tone(-50))).toBeCloseTo(0.5, 1);
    expect(vad.probability(tone(-44))).toBeCloseTo(1, 1);
  });

  it('forgets the floor on reset', () => {
    const vad = new EnergyVad();
    for (let i = 0; i < framesFor(1); i++) vad.probability(tone(-20, i));
    vad.reset();
    vad.probability(new Float32Array(FRAME));
    expect(vad.probability(tone(-20))).toBe(1);
  });
});
