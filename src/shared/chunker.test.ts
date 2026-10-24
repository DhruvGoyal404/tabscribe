import { describe, expect, it } from 'vitest';
import { Chunker } from './chunker';

const ramp = (from: number, length: number) => Float32Array.from({ length }, (_, i) => from + i);

describe('Chunker', () => {
  it('returns nothing until a chunk fills', () => {
    const chunker = new Chunker(4);
    expect(chunker.push(ramp(0, 3))).toEqual([]);
    expect(chunker.push(ramp(3, 1)).map((chunk) => [...chunk])).toEqual([[0, 1, 2, 3]]);
  });

  it('splits input across chunks and keeps the rest', () => {
    const chunker = new Chunker(4);
    expect(chunker.push(ramp(0, 10)).map((chunk) => [...chunk])).toEqual([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
    ]);
    expect(chunker.push(ramp(10, 2)).map((chunk) => [...chunk])).toEqual([[8, 9, 10, 11]]);
  });

  it('hands out a fresh buffer each time, so chunks can be transferred', () => {
    const chunker = new Chunker(2);
    const [first, second] = chunker.push(ramp(0, 4));
    expect(first?.buffer).not.toBe(second?.buffer);
  });

  it('copies its input, which the audio thread reuses', () => {
    const chunker = new Chunker(4);
    const input = ramp(0, 2);
    chunker.push(input);
    input.fill(9);
    expect(chunker.push(ramp(2, 2)).map((chunk) => [...chunk])).toEqual([[0, 1, 2, 3]]);
  });
});
