import { describe, expect, it } from 'vitest';
import { defaultThreads, isThreadCount, MAX_THREADS } from './threads';

describe('defaultThreads', () => {
  it('uses one thread without cross-origin isolation', () => {
    expect(defaultThreads(false, 12)).toBe(1);
  });

  it('uses half the cores, at most 4', () => {
    expect(defaultThreads(true, 2)).toBe(1);
    expect(defaultThreads(true, 4)).toBe(2);
    expect(defaultThreads(true, 6)).toBe(3);
    expect(defaultThreads(true, 8)).toBe(4);
    expect(defaultThreads(true, 12)).toBe(4);
    expect(defaultThreads(true, 64)).toBe(4);
  });

  it('never goes below one thread', () => {
    expect(defaultThreads(true, 1)).toBe(1);
    expect(defaultThreads(true, 0)).toBe(1);
  });
});

describe('isThreadCount', () => {
  it('accepts whole numbers from 1 to the maximum', () => {
    expect(isThreadCount(1)).toBe(true);
    expect(isThreadCount(MAX_THREADS)).toBe(true);
  });

  it.each([0, -1, MAX_THREADS + 1, 1.5, NaN, '4', null])('rejects %s', (value) => {
    expect(isThreadCount(value)).toBe(false);
  });
});
