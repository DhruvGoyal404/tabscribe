// Word and character error rates with text normalisation, for scoring ASR output against
// reference transcripts. Normalisation is applied identically to both sides, so it only has to
// make equivalent spellings compare equal, not produce "correct" text.

export interface ErrorCounts {
  substitutions: number;
  deletions: number;
  insertions: number;
  /** Tokens in the reference. */
  referenceLength: number;
  /** (S + D + I) / N. Infinity when the reference is empty but the hypothesis isn't. */
  rate: number;
}

const ONES = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
  'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
  'nineteen',
]; // prettier-ignore
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const SCALES: [number, string][] = [
  [1e12, 'trillion'],
  [1e9, 'billion'],
  [1e6, 'million'],
  [1e3, 'thousand'],
  [100, 'hundred'],
];
const ORDINAL_WORDS: Record<string, string> = {
  one: 'first', two: 'second', three: 'third', five: 'fifth', eight: 'eighth', nine: 'ninth',
  twelve: 'twelfth',
}; // prettier-ignore

/** Cardinal English words for a non-negative integer, e.g. 1965 -> "one thousand nine hundred sixty five". */
export function integerToWords(value: number): string {
  if (!Number.isSafeInteger(value) || value < 0)
    throw new RangeError(`Not a count: ${String(value)}`);
  if (value < 20) return ONES[value] ?? '';
  if (value < 100) {
    const tens = TENS[Math.floor(value / 10)] ?? '';
    return value % 10 === 0 ? tens : `${tens} ${ONES[value % 10] ?? ''}`;
  }
  for (const [scale, name] of SCALES) {
    if (value >= scale) {
      const head = `${integerToWords(Math.floor(value / scale))} ${name}`;
      const rest = value % scale;
      return rest === 0 ? head : `${head} ${integerToWords(rest)}`;
    }
  }
  return '';
}

function ordinalWords(value: number): string {
  const words = integerToWords(value).split(' ');
  const last = words.pop() ?? '';
  const ordinal =
    ORDINAL_WORDS[last] ?? (last.endsWith('y') ? `${last.slice(0, -1)}ieth` : `${last}th`);
  return [...words, ordinal].join(' ');
}

/** Spells out English numbers: "1,000" "3.5" "21st" "50%". Very long digit runs are read digit by digit. */
function spellEnglishNumbers(text: string): string {
  return text
    .replace(/(\d),(?=\d{3}\b)/g, '$1')
    .replace(/(\d+)%/g, '$1 percent')
    .replace(/\b(\d+)(st|nd|rd|th)\b/g, (_, digits: string) => ordinalWords(Number(digits)))
    .replace(/\b(\d+)\.(\d+)\b/g, (_, whole: string, fraction: string) =>
      [
        integerToWords(Number(whole)),
        'point',
        ...Array.from(fraction, (d) => ONES[Number(d)]),
      ].join(' '),
    )
    .replace(/\d+/g, (digits) =>
      digits.length > 15
        ? Array.from(digits, (d) => ONES[Number(d)]).join(' ')
        : ` ${integerToWords(Number(digits))} `,
    );
}

/** Titles Whisper abbreviates but LibriSpeech-style references spell out. */
const ENGLISH_CANONICAL: Record<string, string> = {
  mister: 'mr',
  missus: 'mrs',
  misses: 'mrs',
  doctor: 'dr',
};

const DEVANAGARI_DIGIT_ZERO = 0x0966;

/**
 * Normalises text for scoring:
 * - Unicode NFKC, then lower case.
 * - Drops bracketed annotations such as "[Music]" or "<|en|>"; keeps the words inside (round)
 *   brackets.
 * - Removes apostrophes ("don't" -> "dont"), then turns all other punctuation and symbols into
 *   spaces. Combining marks are kept, so Devanagari vowel signs and nukta survive.
 * - Devanagari digits become ASCII digits.
 * - English only: numbers are spelled out, and titles get one spelling ("mister" -> "mr").
 */
export function normalizeText(text: string, language: string): string {
  let s = text.normalize('NFKC').toLowerCase();
  s = s.replace(/\[[^\]]*\]|<[^>]*>/g, ' ');
  s = s.replace(/[०-९]/g, (d) => String(d.charCodeAt(0) - DEVANAGARI_DIGIT_ZERO));
  if (language === 'en') s = spellEnglishNumbers(s);
  s = s.replace(/['’‘`]/g, '');
  s = s.replace(/[\p{P}\p{S}]/gu, ' ');
  const words = s.split(/\s+/).filter(Boolean);
  return (language === 'en' ? words.map((w) => ENGLISH_CANONICAL[w] ?? w) : words).join(' ');
}

/** Levenshtein alignment counts. Ties prefer substitution, then deletion, then insertion. */
export function editCounts<T>(reference: readonly T[], hypothesis: readonly T[]): ErrorCounts {
  const n = reference.length;
  const m = hypothesis.length;
  const width = m + 1;
  const cost = new Uint32Array((n + 1) * width);
  // 0 match, 1 substitution, 2 deletion, 3 insertion
  const op = new Uint8Array((n + 1) * width);
  for (let i = 1; i <= n; i++) {
    cost[i * width] = i;
    op[i * width] = 2;
  }
  for (let j = 1; j <= m; j++) {
    cost[j] = j;
    op[j] = 3;
  }
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const same = reference[i - 1] === hypothesis[j - 1];
      const diagonal = (cost[(i - 1) * width + j - 1] ?? 0) + (same ? 0 : 1);
      const deletion = (cost[(i - 1) * width + j] ?? 0) + 1;
      const insertion = (cost[i * width + j - 1] ?? 0) + 1;
      let best = diagonal;
      let bestOp = same ? 0 : 1;
      if (deletion < best) [best, bestOp] = [deletion, 2];
      if (insertion < best) [best, bestOp] = [insertion, 3];
      cost[i * width + j] = best;
      op[i * width + j] = bestOp;
    }
  }
  const counts = { substitutions: 0, deletions: 0, insertions: 0 };
  for (let i = n, j = m; i > 0 || j > 0;) {
    switch (op[i * width + j]) {
      case 1:
        counts.substitutions++;
        i--;
        j--;
        break;
      case 2:
        counts.deletions++;
        i--;
        break;
      case 3:
        counts.insertions++;
        j--;
        break;
      default:
        i--;
        j--;
    }
  }
  const errors = counts.substitutions + counts.deletions + counts.insertions;
  return {
    ...counts,
    referenceLength: n,
    rate: n === 0 ? (errors === 0 ? 0 : Infinity) : errors / n,
  };
}

/** Word error rate after normalisation. */
export function wordErrorRate(
  reference: string,
  hypothesis: string,
  language: string,
): ErrorCounts {
  const words = (text: string) => normalizeText(text, language).split(' ').filter(Boolean);
  return editCounts(words(reference), words(hypothesis));
}

/**
 * Character error rate after normalisation, over Unicode code points with spaces removed, so
 * word-segmentation differences (common in Hindi) don't count twice.
 */
export function characterErrorRate(
  reference: string,
  hypothesis: string,
  language: string,
): ErrorCounts {
  const chars = (text: string) => Array.from(normalizeText(text, language).replace(/ /g, ''));
  return editCounts(chars(reference), chars(hypothesis));
}
