# ASR bench fixtures

Openly licensed clips, used only by the dev-only bench page (`npm run build:bench`). They are
never part of the production build (`check:dist` fails on any `.flac`). The reference transcripts
are in [`../fixtures.ts`](../fixtures.ts), exactly as published with each dataset.

| File                   | Length  | Use                                        | SHA-256                                                            |
| ---------------------- | ------- | ------------------------------------------ | ------------------------------------------------------------------ |
| `en-librispeech.flac`  | 47.7 s  | English read speech, natural pauses        | `2f5d12b597808b13e51ca4fa6608bb47f68eecd7e8734815ea9d84b8940c05e8` |
| `hi-fleurs.flac`       | 50.7 s  | Hindi: nothing should be shown             | `b6ad941a56f89d9b66e87ebcac62bb521644dfd9f55522b583a6d492cf432844` |
| `en-voxpopuli.flac`    | 126.5 s | English, continuous: real forced cuts      | `c8a5fb229f0b37fcd4e8b2d30788d2a82e94c55a1b2d623f5c977f0466d1b911` |
| `fr-de-fleurs.flac`    | 63.3 s  | French and German: nothing should be shown | `3aa14a550f58332d9291e8ca6e39e34aaceb27b5d5c9b0fb79d6a9081b203691` |
| `nonspeech-musan.flac` | 50.0 s  | Music, silence, noise: nothing shown       | `8beda5b838de5af2fe127becff465f233c8f0cf5b823cd2efdb61cd0704ffc15` |

All are FLAC, 16 kHz, mono, 16-bit. Every source was already 16 kHz, so nothing was resampled.
VoxPopuli's audio is 32-bit float and was rounded to 16-bit; the others were 16-bit already.

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

## `en-voxpopuli.flac`: continuous English

- **Source:** VoxPopuli (Wang et al., 2021), `facebook/voxpopuli` at revision
  `42f01879c780b4a2e90ec0b4f616c2ece526e4f1`, config `en`, `validation` split: European Parliament
  plenary speeches.
- **Licence:** [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), per the dataset card
  (see also the [European Parliament's legal notice](https://www.europarl.europa.eu/legal-notice/en/)
  for the raw recordings).
- **Content:** four utterances by four speakers (two men, two women), each a long run with few
  pauses: at the default 10 s maximum the segmenter cuts them 14 times. In order, separated by
  0.6 s of silence:

  | `audio_id`                                         | Speaker | Length |
  | -------------------------------------------------- | ------- | ------ |
  | `20150910-0900-PLENARY-6-en_20150910-11:06:41_2`   | 119435  | 33.3 s |
  | `20180912-0900-PLENARY-en_20180912-21:30:55_5`     | 96776   | 34.7 s |
  | `20121022-0900-PLENARY-15-en_20121022-18:49:22_20` | 97050   | 27.1 s |
  | `20180118-0900-PLENARY-4-en_20180118-10:27:06_3`   | 96812   | 29.5 s |

- **Reference:** each utterance's `raw_text` (all four have `is_gold_transcript` true).

## `fr-de-fleurs.flac`: French and German

- **Source:** FLEURS, `google/fleurs` at revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`,
  configs `fr_fr` and `de_de`, `dev` split.
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), per the dataset card.
- **Content:** three French then three German sentences (all male speakers), 8–13 s each, without
  digits, separated by 0.4 s of silence: French FLoRes ids 1529, 1543, 1550
  (`11000058133089524989.wav`, `10864361718248331558.wav`, `11268749000541560281.wav`), German
  1616, 1581, 1536 (`10271331673746997728.wav`, `10249467818655484680.wav`,
  `10495200964727225611.wav`).
- **Reference:** the FLEURS `raw_transcription` of each.

## `nonspeech-musan.flac`: music, silence, noise

- **Source:** MUSAN (Snyder, Chen and Povey, 2015), <https://www.openslr.org/17>. The music file
  was taken from the `corypaik/musan` mirror (revision `76f9882cfa4475efe11508ac9aa32722f84ca5b7`,
  `music/train-00001-of-00010.parquet`), the noise from `FluidInference/musan` (revision
  `3edcfdf89b56dbe6a395ff29f9c29489e03d1321`). Both mirror MUSAN's files and LICENSE files
  unchanged (SHA-256 of the source WAVs: music `c58ec872…0f671b62`, noise `be147d20…2083a764`).
- **Content:** 0–20 s: `music/fma/music-fma-0098.wav` from 60 s to 80 s, "No Tears, No Rust" by
  Quiet Music for Tiny Robots, instrumental (MUSAN's ANNOTATIONS: no vocals). 20–30 s: digital
  silence. 30–50 s: the first 20 s of `noise/free-sound/noise-free-sound-0232.wav`, listed by MUSAN
  as a background noise.
- **Licences:** the music is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), "No Tears,
  No Rust" by Quiet Music for Tiny Robots, from
  <http://freemusicarchive.org/music/Quiet_Music_for_Tiny_Robots/The_February_Album> (per MUSAN's
  `music/fma/LICENSE`); this clip is an excerpt. The noise is in the public domain (MUSAN's
  `noise/free-sound/LICENSE`: "All selected recordings were marked as in the Public Domain on Free
  Sound").
- **Reference:** none: captions should show nothing.

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
# VoxPopuli: the four utterances (vp1-4.wav, in the order above) with 0.6 s of silence between
ffmpeg -i vp1.wav -i vp2.wav -i vp3.wav -i vp4.wav -f lavfi -t 0.6 -i anullsrc=r=16000:cl=mono \
  -filter_complex "[4]asplit=3[s1][s2][s3];[0][s1][1][s2][2][s3][3]concat=n=7:v=0:a=1[a]" \
  -map "[a]" -ac 1 -ar 16000 -sample_fmt s16 -c:a flac -compression_level 12 -map_metadata -1 \
  en-voxpopuli.flac
# FLEURS fr/de: the same as Hindi, with the six clips above (fr1-3, de1-3) and 0.4 s gaps
# MUSAN: 20 s of the music from 60 s, 10 s of silence, 20 s of the noise
ffmpeg -ss 60 -t 20 -i music-fma-0098.wav -f lavfi -t 10 -i anullsrc=r=16000:cl=mono \
  -t 20 -i noise-free-sound-0232.wav -filter_complex "[0][1][2]concat=n=3:v=0:a=1[a]" \
  -map "[a]" -ac 1 -ar 16000 -sample_fmt s16 -c:a flac -compression_level 12 -map_metadata -1 \
  nonspeech-musan.flac
```
