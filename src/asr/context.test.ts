import { describe, expect, it } from 'vitest';
import {
  dropRepeatedPrefix,
  LONG_SILENCE_SAMPLES,
  PROMPT_TOKENS,
  PromptContext,
  type DecodedSegment,
} from './context';

const segment = (overrides: Partial<DecodedSegment> = {}): DecodedSegment => ({
  startSample: 0,
  endSample: 160_000,
  reason: 'max-length',
  verdict: 'ok',
  language: 'en',
  tokens: Array.from({ length: 40 }, (_, i) => 1000 + i),
  ...overrides,
});

describe('PromptContext', () => {
  it('prompts with the last tokens of a forced cut that passed every guard', () => {
    const context = new PromptContext();
    context.record(segment());
    const prompt = context.promptFor(128_000);
    expect(prompt?.language).toBe('en');
    expect(prompt?.tokens).toHaveLength(PROMPT_TOKENS);
    expect(prompt?.tokens.at(-1)).toBe(1039);
  });

  it('has nothing to prompt with at first', () => {
    expect(new PromptContext().promptFor(0)).toBeNull();
  });

  it('never prompts after a natural pause or the end of the stream', () => {
    for (const reason of ['silence', 'flush'] as const) {
      const context = new PromptContext();
      context.record(segment({ reason }));
      expect(context.promptFor(170_000)).toBeNull();
    }
  });

  it.each(['no-speech', 'loop', 'low-confidence', 'not-english'] as const)(
    'never prompts after a %s decode',
    (verdict) => {
      const context = new PromptContext();
      context.record(segment({ verdict }));
      expect(context.promptFor(128_000)).toBeNull();
    },
  );

  it('never prompts with an empty decode', () => {
    const context = new PromptContext();
    context.record(segment({ tokens: [] }));
    expect(context.promptFor(128_000)).toBeNull();
  });

  it('forgets everything after a long silence', () => {
    const context = new PromptContext();
    context.record(segment());
    expect(context.promptFor(160_000 + LONG_SILENCE_SAMPLES + 1)).toBeNull();
    // Still gone for a later segment that would otherwise qualify.
    expect(context.promptFor(128_000)).toBeNull();
  });

  it('keeps short prompts whole', () => {
    const context = new PromptContext();
    context.record(segment({ tokens: [7, 8, 9] }));
    expect(context.promptFor(128_000)?.tokens).toEqual([7, 8, 9]);
  });

  it('only remembers the latest segment', () => {
    const context = new PromptContext();
    context.record(segment());
    context.record(segment({ reason: 'silence' }));
    expect(context.promptFor(170_000)).toBeNull();
  });
});

describe('dropRepeatedPrefix', () => {
  it('drops the words the overlap repeated', () => {
    expect(
      dropRepeatedPrefix(
        'He has graved doubts whether Sir Frederick',
        'Sir Frederick Layton’s work is really Greek.',
      ),
    ).toBe('Layton’s work is really Greek.');
  });

  it('ignores case and punctuation when matching', () => {
    expect(dropRepeatedPrefix('of the middle classes,', 'Middle classes and we are glad.')).toBe(
      'and we are glad.',
    );
  });

  it('prefers the longest repeat', () => {
    expect(dropRepeatedPrefix('to the mind to the mind', 'to the mind. He has')).toBe('He has');
  });

  it('keeps a lone short word, but drops a lone long one', () => {
    expect(dropRepeatedPrefix('one of the', 'the general principles')).toBe(
      'the general principles',
    );
    expect(dropRepeatedPrefix('his gospel', 'Gospel. Nor is')).toBe('Nor is');
  });

  it('skips lone punctuation between repeated words', () => {
    expect(dropRepeatedPrefix('is really Greek', 'really -- Greek after all.')).toBe('after all.');
  });

  it('leaves text without a repeat alone', () => {
    expect(dropRepeatedPrefix('He has graved doubts', '  Under general principles. ')).toBe(
      'Under general principles.',
    );
    expect(dropRepeatedPrefix('', 'Hello there.')).toBe('Hello there.');
    expect(dropRepeatedPrefix('Hello there.', '')).toBe('');
  });

  it('allows one word in four to be decoded differently', () => {
    expect(
      dropRepeatedPrefix(
        'how luminous these criticisms are. how delicate and expression',
        'how delicate an expression On the general principles',
      ),
    ).toBe('On the general principles');
  });

  it('allows the word at the cut to be garbled on either side', () => {
    // The cut garbled the previous segment's last word ("you" for "his").
    expect(
      dropRepeatedPrefix(
        'and we are glad to welcome you.',
        'and we are glad to welcome his gospel.',
      ),
    ).toBe('his gospel.');
    // The next segment starts with a fragment of the word before the overlap.
    expect(
      dropRepeatedPrefix(
        'at this festive season of the year,',
        'of season of the year with Christmas',
      ),
    ).toBe('with Christmas');
  });

  it('needs an exact match for short repeats', () => {
    expect(dropRepeatedPrefix('to welcome him', 'to welcome his gospel')).toBe(
      'to welcome his gospel',
    );
    expect(dropRepeatedPrefix('glad to see', 'we glad to meet')).toBe('we glad to meet');
  });

  it('does not match unrelated text', () => {
    expect(
      dropRepeatedPrefix('He has grave doubts whether', 'It is obviously unnecessary for us'),
    ).toBe('It is obviously unnecessary for us');
  });

  it('drops the whole segment when it only repeats', () => {
    expect(dropRepeatedPrefix('and can discover in it', 'discover in it')).toBe('');
  });
});
