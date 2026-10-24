import { describe, expect, it } from 'vitest';
import { compressionRatio, hasLoop, judge, maxNewTokens, words, type Attempt } from './guards';

const attempt = (overrides: Partial<Attempt> = {}): Attempt => ({
  text: 'Mr. Quilter is the apostle of the middle classes.',
  temperature: 0,
  avgLogprob: -0.3,
  compressionRatio: 1.1,
  hitTokenCap: false,
  ...overrides,
});

describe('maxNewTokens', () => {
  it('allows 8 tokens per second plus 8, at most 224', () => {
    expect(maxNewTokens(1)).toBe(16);
    expect(maxNewTokens(10)).toBe(88);
    expect(maxNewTokens(2.01)).toBe(25);
    expect(maxNewTokens(30)).toBe(224);
  });
});

describe('compressionRatio', () => {
  it('is low for ordinary speech and high for a loop', async () => {
    const speech =
      'Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. ' +
      "Nor is Mr. Quilter's manner less interesting than his matter.";
    expect(await compressionRatio(speech)).toBeLessThan(2.4);
    expect(await compressionRatio('Thank you for watching. '.repeat(8))).toBeGreaterThan(2.4);
  });

  it('is 0 for empty text', async () => {
    expect(await compressionRatio('')).toBe(0);
  });
});

describe('words', () => {
  it('lower-cases and strips punctuation, keeping apostrophes', () => {
    expect(words("  Nor is Mr. Quilter's manner -- less? ")).toEqual([
      'nor',
      'is',
      'mr',
      "quilter's",
      'manner',
      'less',
    ]);
  });
});

describe('hasLoop', () => {
  it.each([
    'no no no no',
    'I think, I think, I think so.',
    'and then we went and then we went and then we went home',
    'a b c d e f g h a b c d e f g h a b c d e f g h',
  ])('finds the loop in %j', (text) => {
    expect(hasLoop(text)).toBe(true);
  });

  it.each([
    'no no no',
    'I think, I think so.',
    'very, very good',
    'Mr. Quilter is the apostle of the middle classes.',
    '',
  ])('accepts %j', (text) => {
    expect(hasLoop(text)).toBe(false);
  });
});

describe('judge', () => {
  it('passes a confident, varied decode', () => {
    expect(judge(attempt(), 0.01)).toBe('ok');
  });

  it('calls a doubtful decode of likely silence no-speech, before any loop check', () => {
    expect(judge(attempt({ avgLogprob: -1.2 }), 0.7)).toBe('no-speech');
    expect(judge(attempt({ avgLogprob: -1.2, compressionRatio: 3 }), 0.7)).toBe('no-speech');
  });

  it('keeps a confident decode even when no-speech is likely', () => {
    expect(judge(attempt(), 0.9)).toBe('ok');
  });

  it('flags loops by compression ratio, token cap or repeated phrases', () => {
    expect(judge(attempt({ compressionRatio: 2.5 }), 0)).toBe('loop');
    expect(judge(attempt({ hitTokenCap: true }), 0)).toBe('loop');
    expect(judge(attempt({ text: 'thank you thank you thank you' }), 0)).toBe('loop');
  });

  it('flags a doubtful decode that is not silence as low-confidence', () => {
    expect(judge(attempt({ avgLogprob: -1.5 }), 0.2)).toBe('low-confidence');
  });
});
