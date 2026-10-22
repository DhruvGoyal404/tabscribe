import { describe, expect, it } from 'vitest';
import {
  characterErrorRate,
  editCounts,
  integerToWords,
  normalizeText,
  wordErrorRate,
} from './metrics';

describe('integerToWords', () => {
  it.each([
    [0, 'zero'],
    [7, 'seven'],
    [13, 'thirteen'],
    [20, 'twenty'],
    [42, 'forty two'],
    [100, 'one hundred'],
    [105, 'one hundred five'],
    [1965, 'one thousand nine hundred sixty five'],
    [2000, 'two thousand'],
    [1_000_001, 'one million one'],
  ])('%i -> %s', (value, words) => {
    expect(integerToWords(value)).toBe(words);
  });

  it('rejects non-counts', () => {
    expect(() => integerToWords(-1)).toThrow(RangeError);
    expect(() => integerToWords(1.5)).toThrow(RangeError);
  });
});

describe('normalizeText', () => {
  it.each([
    ['case and punctuation', 'Hello, World!', 'hello world'],
    ['apostrophes join', "Don't stop; it's fine.", 'dont stop its fine'],
    ['curly apostrophes', 'Quilter’s manner', 'quilters manner'],
    ['hyphens split', 'well-known up-to-date', 'well known up to date'],
    ['bracketed annotations', '[Music] Hello <|en|> there', 'hello there'],
    ['round brackets keep words', 'find animals (prey) by smell', 'find animals prey by smell'],
    ['whitespace', '  a \n\t b  ', 'a b'],
    ['titles', 'Mr. Quilter met Mister Ruskin and Dr. Who', 'mr quilter met mr ruskin and dr who'],
  ])('English: %s', (_label, input, expected) => {
    expect(normalizeText(input, 'en')).toBe(expected);
  });

  it.each([
    ['plain numbers', 'It cost 42 pounds', 'it cost forty two pounds'],
    ['thousands separators', '1,000 people', 'one thousand people'],
    ['decimals', '3.25 metres', 'three point two five metres'],
    ['percentages', '50% off', 'fifty percent off'],
    ['ordinals', 'the 21st and 2nd and 3rd', 'the twenty first and second and third'],
    ['ordinal tens', 'his 40th', 'his fortieth'],
    ['digits glued to words', 'MP3 player', 'mp three player'],
  ])('English numbers: %s', (_label, input, expected) => {
    expect(normalizeText(input, 'en')).toBe(expected);
  });

  it('keeps Devanagari combining marks and strips the danda', () => {
    expect(normalizeText('कुतूहल गाँव में आधे घंटे टहलना समय की हानि नहीं है।', 'hi')).toBe(
      'कुतूहल गाँव में आधे घंटे टहलना समय की हानि नहीं है',
    );
  });

  it('maps Devanagari digits to ASCII and leaves Hindi numbers as digits', () => {
    expect(normalizeText('११ चौके और 2 छक्के', 'hi')).toBe('11 चौके और 2 छक्के');
  });

  it('treats precomposed and decomposed nukta letters the same', () => {
    expect(normalizeText('ज़', 'hi')).toBe(normalizeText('ज़', 'hi'));
  });
});

describe('editCounts', () => {
  it('counts substitutions, deletions and insertions', () => {
    expect(editCounts(['a', 'b', 'c', 'd'], ['a', 'x', 'c'])).toEqual({
      substitutions: 1,
      deletions: 1,
      insertions: 0,
      referenceLength: 4,
      rate: 0.5,
    });
    expect(editCounts(['a'], ['a', 'b', 'c'])).toMatchObject({ insertions: 2, rate: 2 });
  });

  it('handles empty sides', () => {
    expect(editCounts([], [])).toMatchObject({ rate: 0, referenceLength: 0 });
    expect(editCounts([], ['a']).rate).toBe(Infinity);
    expect(editCounts(['a', 'b'], [])).toMatchObject({ deletions: 2, rate: 1 });
  });
});

describe('wordErrorRate', () => {
  it('is zero for a perfect transcript in a different style', () => {
    const reference =
      "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER";
    const hypothesis =
      "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter's manner less interesting than his matter.";
    expect(wordErrorRate(reference, hypothesis, 'en').rate).toBe(0);
  });

  it('scores real mistakes', () => {
    const result = wordErrorRate('the cat sat on the mat', 'the cat sat on a mat today', 'en');
    expect(result).toMatchObject({ substitutions: 1, insertions: 1, referenceLength: 6 });
    expect(result.rate).toBeCloseTo(2 / 6);
  });

  it('works for Hindi words', () => {
    const reference = 'कुतूहल गाँव में आधे घंटे टहलना समय की हानि नहीं है।';
    expect(
      wordErrorRate(reference, 'कुतूहल गांव में आधे घंटे टहलना समय की हानि नहीं है', 'hi'),
    ).toMatchObject({
      substitutions: 1,
      referenceLength: 11,
    });
  });
});

describe('characterErrorRate', () => {
  it('ignores word segmentation and counts code points', () => {
    expect(characterErrorRate('समय की हानि', 'समयकी हानि', 'hi').rate).toBe(0);
    // गाँव vs गांव: candrabindu (U+0901) vs anusvara (U+0902) is one substitution.
    expect(characterErrorRate('गाँव', 'गांव', 'hi')).toMatchObject({
      substitutions: 1,
      referenceLength: 4,
    });
  });
});
