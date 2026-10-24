// Checks on a Whisper decode, after openai/whisper's transcribe(): a decode that compresses too well
// (a loop) or that the model itself doubts (low average log-probability) is retried at a higher
// temperature, and audio the model calls silence is dropped. Pure, apart from the compression
// ratio, which uses the platform's deflate.

/** openai/whisper's thresholds. */
export const COMPRESSION_RATIO_THRESHOLD = 2.4;
export const LOGPROB_THRESHOLD = -1.0;
export const NO_SPEECH_THRESHOLD = 0.6;
/** Greedy first, then two sampled retries (openai/whisper goes on to 1.0 in 0.2 steps). */
export const TEMPERATURES = [0, 0.2, 0.4] as const;
/** Bans any repeated 4-token sequence among the generated tokens. */
export const NO_REPEAT_NGRAM_SIZE = 4;

/** Whisper decodes at most 224 new tokens per window. */
const MAX_NEW_TOKENS = 224;
/** Fast English speech is about 5.5 tokens per second; more than this per second is a loop. */
const TOKENS_PER_SECOND = 8;
const TOKEN_ALLOWANCE = 8;

/** The token cap for `seconds` of audio: well above real speech, well below a runaway loop. */
export const maxNewTokens = (seconds: number): number =>
  Math.min(MAX_NEW_TOKENS, Math.ceil(seconds * TOKENS_PER_SECOND) + TOKEN_ALLOWANCE);

/** UTF-8 bytes over zlib-compressed bytes, as openai/whisper computes it. */
export async function compressionRatio(text: string): Promise<number> {
  const bytes = new TextEncoder().encode(text);
  if (bytes.length === 0) return 0;
  const compressed = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  const size = (await new Response(compressed).arrayBuffer()).byteLength;
  return bytes.length / size;
}

const normaliseWord = (word: string): string => word.toLowerCase().replace(/[^\p{L}\p{N}']/gu, '');

/** Lower-case words without punctuation; empty words (lone punctuation) are dropped. */
export const words = (text: string): string[] =>
  text.split(/\s+/).map(normaliseWord).filter(Boolean);

/** Longest phrase checked for back-to-back repeats. */
const MAX_LOOP_PHRASE = 8;

/**
 * Whether some phrase repeats back to back: one word four times ("no no no no"), or a longer phrase
 * three times. Short texts can loop without compressing well, so this complements the ratio.
 */
export function hasLoop(text: string): boolean {
  const list = words(text);
  for (let size = 1; size <= MAX_LOOP_PHRASE; size++) {
    const needed = size === 1 ? 4 : 3;
    for (let start = 0; start + size * needed <= list.length; start++) {
      let repeats = 1;
      while (
        repeats < needed &&
        list
          .slice(start, start + size)
          .every((word, i) => word === list[start + repeats * size + i])
      ) {
        repeats += 1;
      }
      if (repeats >= needed) return true;
    }
  }
  return false;
}

export interface Attempt {
  text: string;
  temperature: number;
  /** Mean log-probability of the generated tokens, end-of-text included. */
  avgLogprob: number;
  compressionRatio: number;
  /** The decode stopped at the token cap instead of at end-of-text. */
  hitTokenCap: boolean;
}

export type Verdict =
  /** Passed every check. */
  | 'ok'
  /** The model calls it silence and isn't confident in its text: drop it. */
  | 'no-speech'
  /** Retry hotter; on the last attempt, this becomes the outcome. */
  | 'loop'
  | 'low-confidence';

export function judge(attempt: Attempt, noSpeechProb: number): Verdict {
  const doubtful = attempt.avgLogprob < LOGPROB_THRESHOLD;
  if (noSpeechProb > NO_SPEECH_THRESHOLD && doubtful) return 'no-speech';
  if (
    attempt.compressionRatio > COMPRESSION_RATIO_THRESHOLD ||
    attempt.hitTokenCap ||
    hasLoop(attempt.text)
  ) {
    return 'loop';
  }
  return doubtful ? 'low-confidence' : 'ok';
}
