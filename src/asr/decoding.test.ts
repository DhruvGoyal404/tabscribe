import { Tensor } from '@huggingface/transformers';
import { describe, expect, it } from 'vitest';
import { GeneratedNoRepeatNGram, Interrupt, logProbability, TokenLogprobs } from './decoding';

const logits = (values: number[]): Tensor =>
  new Tensor('float32', Float32Array.from(values), [1, values.length]);
const zeros = (count: number): Tensor => logits(new Array<number>(count).fill(0));
const ids = (...tokens: number[]): bigint[][] => [tokens.map(BigInt)];
const banned = (tensor: Tensor): number[] =>
  [...(tensor.data as Float32Array)].flatMap((score, token) =>
    score === -Infinity ? [token] : [],
  );

describe('GeneratedNoRepeatNGram', () => {
  it('bans the token that would repeat a generated n-gram', () => {
    // Generated 1 2 3 1 2: "1 2 3" was seen, so 3 can't follow "1 2" again.
    const processor = new GeneratedNoRepeatNGram(3, 2);
    expect(banned(processor._call(ids(90, 91, 1, 2, 3, 1, 2), zeros(8)))).toEqual([3]);
  });

  it('ignores n-grams in the prompt', () => {
    // "1 2 3" only occurs in the prompt (first 5 tokens).
    const processor = new GeneratedNoRepeatNGram(3, 5);
    expect(banned(processor._call(ids(1, 2, 3, 9, 9, 1, 2), zeros(8)))).toEqual([]);
  });

  it('bans every token that followed the context', () => {
    const processor = new GeneratedNoRepeatNGram(2, 0);
    expect(banned(processor._call(ids(5, 1, 5, 2, 5), zeros(8)))).toEqual([1, 2]);
  });

  it('does nothing before enough tokens exist', () => {
    const processor = new GeneratedNoRepeatNGram(4, 1);
    expect(banned(processor._call(ids(0, 1, 1), zeros(4)))).toEqual([]);
  });
});

describe('logProbability', () => {
  it('is log_softmax at the token', () => {
    const scores = Float32Array.from([1, 2, 3]);
    const expected = 3 - Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3));
    expect(logProbability(scores, 2)).toBeCloseTo(expected, 6);
  });

  it('undoes a temperature with `scale`', () => {
    const raw = Float32Array.from([1, 2, 3]);
    const heated = raw.map((x) => x / 0.4);
    expect(logProbability(heated, 0, 0.4)).toBeCloseTo(logProbability(raw, 0), 5);
  });

  it('treats suppressed tokens as impossible', () => {
    expect(logProbability(Float32Array.from([0, -Infinity]), 0)).toBeCloseTo(0, 6);
    expect(logProbability(Float32Array.from([0, -Infinity]), 1)).toBe(-Infinity);
  });
});

describe('TokenLogprobs', () => {
  it('averages the chosen tokens, reading each step once the token is known', () => {
    const recorder = new TokenLogprobs(0);
    const first = logits([0, 0]);
    recorder(ids(7), first);
    // A later processor edits step 1 after the recorder ran: the recorder must see the edit.
    (first.data as Float32Array)[1] = -Infinity;
    recorder(ids(7, 0), logits([0, 0]));
    const average = recorder.average(1n);
    // Step 1 chose token 0 with probability 1, step 2 chose token 1 with probability 1/2.
    expect(average).toBeCloseTo((0 + Math.log(0.5)) / 2, 6);
  });

  it('is -Infinity with nothing generated', () => {
    expect(new TokenLogprobs(0).average(undefined)).toBe(-Infinity);
  });
});

describe('Interrupt', () => {
  it('stops every sequence once requested', () => {
    const interrupt = new Interrupt();
    expect(interrupt(ids(1, 2))).toEqual([false]);
    interrupt.requested = true;
    expect(interrupt(ids(1, 2))).toEqual([true]);
  });
});
