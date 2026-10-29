// What carries over from one segment to the next.
//
// After a forced cut the next segment starts mid-sentence, and Whisper may skip the clipped opening
// clause (see docs/spikes/p3b-streaming.md). Prompting it with the previous segment's text lets it
// continue the sentence instead. Conditioning on previous text is also how one bad decode turns
// into a run of them, so the prompt is restricted: only after a forced cut, only the last few
// tokens, only from a decode that passed every guard, and never across a long silence or a change
// of language (the worker checks the language, since it detects it).
//
// With an overlap (the segmenter's overlapMs, off by default), the segment after a forced cut
// repeats the end of the one before, so a word split by the cut is heard whole once. Its text then
// starts with words the previous caption already showed, and dropRepeatedPrefix removes them.
import type { EndReason } from '../vad/segmenter';
import { words, type Verdict } from './guards';

/** openai/whisper allows up to 223 prompt tokens; a short tail is enough to continue a sentence. */
export const PROMPT_TOKENS = 24;
/** A pause this long ends any carried-over context. */
export const LONG_SILENCE_SAMPLES = 16_000 * 2;

export interface DecodedSegment {
  startSample: number;
  endSample: number;
  reason: EndReason;
  verdict: Verdict | 'not-english';
  language: string;
  /** Generated text tokens, without special tokens. */
  tokens: readonly number[];
}

export interface Prompt {
  tokens: number[];
  /** The worker drops the prompt when the new segment's language differs. */
  language: string;
}

export class PromptContext {
  private last: DecodedSegment | null = null;

  constructor(
    private readonly maxTokens = PROMPT_TOKENS,
    private readonly longSilenceSamples = LONG_SILENCE_SAMPLES,
  ) {}

  /** The prompt for a segment starting at `startSample`, or null to decode without one. */
  promptFor(startSample: number): Prompt | null {
    const last = this.last;
    if (!last) return null;
    if (startSample - last.endSample > this.longSilenceSamples) {
      this.reset();
      return null;
    }
    if (last.reason !== 'max-length' || last.verdict !== 'ok' || last.tokens.length === 0) {
      return null;
    }
    return { tokens: last.tokens.slice(-this.maxTokens), language: last.language };
  }

  record(segment: DecodedSegment): void {
    this.last = segment;
  }

  reset(): void {
    this.last = null;
  }
}

/** Most words a 2 s overlap can repeat. */
const MAX_REPEATED_WORDS = 12;
/** One word in this many may be decoded differently the second time ("and" for "an"). */
const WORDS_PER_MISMATCH = 4;

/**
 * `next` without the leading words that repeat the end of `previous`.
 *
 * The overlap is decoded twice, so the repeat isn't always exact: up to one word in four may
 * differ, and one word at the cut may be garbled on either side (the previous segment's last word,
 * or the next one's first). Only an exact repeat of a single word counts, and only if it's longer
 * than three letters, so a lone "the" or "and" isn't eaten. The match with the most equal words
 * wins, then the one with fewer differences.
 */
export function dropRepeatedPrefix(previous: string, next: string): string {
  const before = words(previous);
  const raw = next.trim().split(/\s+/).filter(Boolean);
  // The words of `next` with where they sit in `raw` (lone punctuation has no word).
  const heads = raw.flatMap((token, index) => words(token).map((word) => ({ word, index })));
  let best: { matched: number; mismatches: number; last: number } | null = null;
  for (let count = Math.min(MAX_REPEATED_WORDS, before.length, heads.length); count >= 1; count--) {
    for (const skipBefore of [0, 1]) {
      for (const skipAfter of [0, 1]) {
        const end = before.length - skipBefore;
        if (end - count < 0 || skipAfter + count > heads.length) continue;
        const exact = skipBefore === 0 && skipAfter === 0;
        if (count === 1 && !(exact && (before[end - 1]?.length ?? 0) > 3)) continue;
        if (!exact && count < 3) continue;
        const tail = before.slice(end - count, end);
        const mismatches = tail.filter((word, i) => word !== heads[skipAfter + i]?.word).length;
        if (mismatches > Math.floor(count / WORDS_PER_MISMATCH)) continue;
        const matched = count - mismatches;
        if (
          !best ||
          matched > best.matched ||
          (matched === best.matched && mismatches < best.mismatches)
        ) {
          best = { matched, mismatches, last: heads[skipAfter + count - 1]?.index ?? -1 };
        }
      }
    }
  }
  return raw.slice(best ? best.last + 1 : 0).join(' ');
}
