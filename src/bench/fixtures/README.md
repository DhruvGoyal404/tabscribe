# ASR bench fixtures

Two short clips of openly licensed read speech, used only by the dev-only bench page
(`npm run build:bench`). They are never part of the production build. The reference transcripts
are in [`../fixtures.ts`](../fixtures.ts), exactly as published with each dataset.

| File                  | Length | Format                     | SHA-256                                                            |
| --------------------- | ------ | -------------------------- | ------------------------------------------------------------------ |
| `en-librispeech.flac` | 47.7 s | FLAC, 16 kHz, mono, 16-bit | `2f5d12b597808b13e51ca4fa6608bb47f68eecd7e8734815ea9d84b8940c05e8` |
| `hi-fleurs.flac`      | 50.7 s | FLAC, 16 kHz, mono, 16-bit | `b6ad941a56f89d9b66e87ebcac62bb521644dfd9f55522b583a6d492cf432844` |

Both were encoded losslessly from the original 16 kHz audio, so nothing was resampled or
re-quantised.

## `en-librispeech.flac`: English

- **Source:** LibriSpeech ASR corpus (Panayotov, Chen, Povey and Khudanpur, 2015),
  <https://www.openslr.org/12>, `dev-clean`, speaker 1272, chapter 128104. Fetched through the
  `hf-internal-testing/librispeech_asr_dummy` mirror (`clean` / `validation`), which carries the
  same audio and transcripts.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The underlying recordings
  are LibriVox public-domain audiobooks.
- **Content:** utterances `1272-128104-0000`, `-0001`, `-0002`, `-0003`, `-0005` and `-0006`,
  concatenated in that order. `-0004` was skipped: at 29 s it alone would be most of the clip.
- **Reference:** the LibriSpeech transcripts (upper case, no punctuation, "MISTER" spelled out).

## `hi-fleurs.flac`: Hindi

- **Source:** FLEURS (Conneau et al., 2022), `google/fleurs` at revision
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, config `hi_in`, `validation` (dev) split.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), per the dataset card. The
  sentences are from FLoRes-101.
- **Content:** five sentences, each a separate FLEURS recording, separated by 0.4 s of silence.
  In order:

  | FLoRes id | FLEURS file                | Length |
  | --------- | -------------------------- | ------ |
  | 1601      | `10028896103651079035.wav` | 10.3 s |
  | 1607      | `10211077472389368581.wav` | 6.4 s  |
  | 1526      | `10429326314841370948.wav` | 14.2 s |
  | 1559      | `10691214664103820058.wav` | 9.2 s  |
  | 1555      | `10905105795491986941.wav` | 9.8 s  |

  Chosen to avoid digits, so CER isn't dominated by numeral-vs-word spelling.

- **Reference:** the FLEURS `raw_transcription` for each sentence.
- **Translation reference:** FLoRes ids are shared across languages, so the English sentences with
  the same ids (from FLEURS `en_us`, `dev.tsv`) are the reference for judging Hindi → English output
  by hand. Whisper's translation isn't expected to match them word for word.

## Rebuilding

With the source clips downloaded as WAV/FLAC into `ls/` and `hi/`:

```sh
# English: concat demuxer over the six LibriSpeech utterances
ffmpeg -f concat -safe 0 -i en.txt -ac 1 -ar 16000 -sample_fmt s16 -c:a flac \
  -compression_level 12 -map_metadata -1 en-librispeech.flac
# Hindi: the five clips with 0.4 s of silence between them
ffmpeg -f lavfi -t 0.4 -i anullsrc=r=16000:cl=mono -sample_fmt s16 sil.wav
ffmpeg -f concat -safe 0 -i hi.txt -ac 1 -ar 16000 -sample_fmt s16 -c:a flac \
  -compression_level 12 -map_metadata -1 hi-fleurs.flac
```
