import { describe, expect, it } from 'vitest';
import type { DecodeOutcome, DecodePrompt } from '../asr/protocol';
import type { Decoded } from '../asr/scheduler';
import { decodeStream, type FinalSegment } from './stream';

const final = (
  segmentId: number,
  startSample: number,
  reason: FinalSegment['reason'],
  overlapSamples = 0,
): FinalSegment => ({
  type: 'final',
  segmentId,
  startSample,
  endSample: startSample + 160_000,
  speechEndSample: startSample + 156_800,
  reason,
  overlapSamples,
  audio: new Float32Array(16),
});

const decoded = (
  text: string,
  outcome: DecodeOutcome = 'ok',
  overrides: Partial<Decoded> = {},
): Decoded => ({
  type: 'decoded',
  id: 0,
  outcome,
  text: outcome === 'ok' || outcome === 'low-confidence' ? text : '',
  tokens: text ? [1, 2, 3] : [],
  language: 'en',
  englishProb: 0.99,
  noSpeechProb: 0.01,
  promptTokens: 0,
  attempts: [
    { temperature: 0, avgLogprob: -0.2, compressionRatio: 1.1, hitTokenCap: false, verdict: 'ok' },
  ],
  featuresMs: 1,
  encoderMs: 1,
  ms: 100,
  ...overrides,
});

/** A fake worker: answers in order, and records the prompts it was given. */
function fake(results: Decoded[]) {
  const prompts: (DecodePrompt | null)[] = [];
  let next = 0;
  const decode = (_audio: Float32Array, prompt: DecodePrompt | null) => {
    prompts.push(prompt);
    const result = results[next++];
    if (!result) throw new Error('No more results');
    // The worker uses a prompt only in the matching language.
    const used = prompt?.language === result.language ? prompt.tokens.length : 0;
    return Promise.resolve({ ...result, promptTokens: used });
  };
  return { decode, prompts };
}

describe('decodeStream', () => {
  it('drops the words an overlap repeats, and joins the captions', async () => {
    const { decode } = fake([
      decoded('He has grave doubts whether Sir Frederick'),
      decoded('Sir Frederick Leighton’s work is really Greek.', 'ok', { englishProb: 0.8 }),
    ]);
    const run = await decodeStream(
      [final(0, 0, 'max-length'), final(1, 128_000, 'silence', 32_000)],
      decode,
      false,
    );
    expect(run.transcript).toBe(
      'He has grave doubts whether Sir Frederick Leighton’s work is really Greek.',
    );
    expect(run.stats.forcedCuts).toBe(1);
    expect(run.stats.englishProbs).toEqual([0.99, 0.8]);
  });

  it('prompts only after a forced cut that passed, and only when asked to', async () => {
    const segments = [
      final(0, 0, 'max-length'),
      final(1, 128_000, 'silence', 32_000),
      final(2, 300_000, 'max-length'),
    ];
    const results = () => [decoded('one two three'), decoded('four five'), decoded('six')];
    const off = fake(results());
    await decodeStream(segments, off.decode, false);
    expect(off.prompts).toEqual([null, null, null]);

    const on = fake(results());
    const run = await decodeStream(segments, on.decode, true);
    expect(on.prompts).toEqual([null, { tokens: [1, 2, 3], language: 'en' }, null]);
    expect(run.stats.prompted).toBe(1);
  });

  it('shows nothing for dropped decodes and never prompts from them', async () => {
    const { decode, prompts } = fake([
      decoded('', 'no-speech'),
      decoded('', 'loop', {
        attempts: [0, 0.2, 0.4].map((temperature) => ({
          temperature,
          avgLogprob: -0.5,
          compressionRatio: 3,
          hitTokenCap: false,
          verdict: 'loop' as const,
        })),
      }),
      decoded('real words'),
    ]);
    const run = await decodeStream(
      [
        final(0, 0, 'max-length'),
        final(1, 128_000, 'max-length', 32_000),
        final(2, 256_000, 'silence', 32_000),
      ],
      decode,
      true,
    );
    expect(prompts).toEqual([null, null, null]);
    expect(run.transcript).toBe('real words');
    expect(run.stats.outcomes).toEqual({ 'no-speech': 1, loop: 1, ok: 1 });
    expect(run.stats.loopsCaught).toBe(3);
    expect(run.stats.retries).toBe(2);
    expect(run.stats.loopsShown).toBe(0);
  });

  it('counts a shown caption that still loops', async () => {
    const { decode } = fake([decoded('thank you thank you thank you', 'low-confidence')]);
    const run = await decodeStream([final(0, 0, 'silence')], decode, false);
    expect(run.stats.loopsShown).toBe(1);
  });

  it('does not use a prompt across a change of language', async () => {
    const { decode } = fake([
      decoded('one two three'),
      decoded('', 'not-english', { language: 'hi' }),
    ]);
    const run = await decodeStream(
      [final(0, 0, 'max-length'), final(1, 128_000, 'silence', 32_000)],
      decode,
      true,
    );
    expect(run.stats.prompted).toBe(0);
    expect(run.captions.map((caption) => caption.result.outcome)).toEqual(['ok', 'not-english']);
  });
});
