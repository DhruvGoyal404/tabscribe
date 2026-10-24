# P3b: streaming English captions

Live English captions from tab audio, built on the P3a decisions (whisper-base: WebGPU fp16/fp16,
fallback WASM q8/q8). Built in PR-sized pieces. This doc grows with each piece.

| Piece | Scope                                                                                      | Status   |
| ----- | ------------------------------------------------------------------------------------------ | -------- |
| 1     | Per-call cost measurement (encoder vs decoder) and the VAD worker                          | PR #9    |
| 1b    | Cross-origin isolation for multithreaded WASM, bench thread selector                       | PR #11   |
| 1c    | Chrome stable results, the q8 step-count check, features for the real audio only           | **this** |
| 2     | ASR side: segment queue, repetition and no-speech guards, interruptible partials, ordering | next     |
| 3     | Wiring: worklet → VAD → ASR in the offscreen document, runtime backend choice, consent     |          |
| 4     | Popup debug transcript, live bench mode (latency p50/p95), regression fixture, memory      |          |

## Pipeline

```text
offscreen document
  AudioWorklet (native rate, mono) ──MessagePort──▶ VAD worker ──MessagePort──▶ ASR worker
                                                   resampler → 16 kHz            segment queue
                                                   Silero VAD (32 ms frames)     guards, Whisper
                                                   segmenter (pauses, max cut)   ordered results
```

- **Three threads, so VAD never waits on Whisper.** A WASM Whisper call blocks its thread for
  seconds. Audio and segments travel over `MessagePort`s with transferred buffers, so nothing
  passes through the offscreen page's main thread.
- **End-of-stream travels on the data path.** Messages on different channels aren't ordered
  relative to each other, so `flush` follows the last audio chunk on the audio port, and `flushed`
  follows the last segment on the segments port.

## Piece 1 results

### Per-call cost (whisper-base, one English decode per call)

Measured with the bench's **Measure per-call cost** panel. Each row is the median of three
segments from different parts of the English clip, after one warm-up call. Features = log-mel
extraction (JS, on the CPU), Encoder = the encoder session, Decoder = greedy decoding from the
encoder output.

**Where:** Chromium 156 under automation, on the same Windows machine (Intel gen-12lp). In the same
session the full benchmark gave EN RTF **0.172** on WebGPU fp16 (your Chrome stable run: 0.188),
so the WebGPU numbers are representative. On WASM q8 it gave **0.560** (your run: 0.266): CPU work
ran about 2× slower under automation, so **the WASM rows need re-measuring on Chrome stable** (see
below).

| Config                 | Segment (s) | Features (ms) | Encoder (ms) | Decoder (ms) | Decoder steps | ms / step | Total (ms) | Call RTF |
| ---------------------- | ----------: | ------------: | -----------: | -----------: | ------------: | --------: | ---------: | -------: |
| webgpu fp16 / fp16     |           2 |           464 |          373 |          670 |             7 |      90.4 |       1471 |    0.735 |
| webgpu fp16 / fp16     |           5 |           446 |          338 |         1735 |            18 |      85.3 |       2392 |    0.478 |
| webgpu fp16 / fp16     |          10 |           470 |          392 |         1903 |            34 |      62.5 |       2949 |    0.295 |
| webgpu fp16 / fp16     |          15 |           322 |          322 |         3803 |            50 |      75.6 |       4449 |    0.297 |
| wasm q8 / q8 (2× slow) |           2 |           314 |         7260 |         1144 |             7 |     157.0 |       9060 |    4.530 |
| wasm q8 / q8 (2× slow) |           5 |           481 |         8433 |         2078 |            18 |     115.6 |      11166 |    2.233 |
| wasm q8 / q8 (2× slow) |          10 |           548 |         7787 |         3282 |            26 |     119.2 |      11569 |    1.157 |
| wasm q8 / q8 (2× slow) |          15 |           466 |         8508 |         4809 |            49 |     100.0 |      13954 |    0.930 |

The decoded text of every call is shown in the panel. It's real and grows with the segment
("Mr. Quilter is the apostle." at 2 s, the full sentence at 10 s).

**Bug found and fixed while measuring:** Whisper's `forward_params` in Transformers.js 4.3.1 omit
`encoder_outputs`, so `generate()` silently discarded a precomputed encoder output and **ran the
encoder a second time**. The first measurement counted a whole extra encoder pass (about 4.5 s on
WASM) as decoder time. The worker now allows `encoder_outputs` on the loaded model, so one encoder
pass can serve the no-speech check and the decode (piece 2).

### What the numbers say

- **WebGPU: the cost is mostly decoding.** The fixed part per call (features plus encoder) is about
  0.7–0.85 s. Decoding costs 60–90 ms per token, and speech produces about 3.4 tokens per second, so
  one call costs roughly **0.8 s + 0.25 s × segment length**.
  - Final-caption latency after an utterance ends ≈ the 500 ms pause + any wait for the worker +
    the call. For typical 3–6 s utterances that's about 2.1–2.8 s, inside the **≤ 3 s p50** target.
  - An **8 s maximum segment** bounds a single call to about 2.8 s.
  - For **p95 ≤ 6 s**, a partial decode in progress must not delay a final by its full length, so
    partials have to be interruptible. Piece 2 stops a partial at the next decoder step (about
    75 ms) when a final is waiting, using a `stopping_criteria` hook.
  - With the first partial at 1 s of speech, a partial costs about 1.1 s, so it shows about 2.1 s
    after speech starts. That's at the ~2 s target; starting at 0.8 s would gain margin.
  - **Feature extraction is 0.3–0.47 s per call,** about a third of the fixed cost on short
    segments. Whisper pads every input to 30 s before computing the log-mel. The padded frames
    reach a known value after the log-mel's clamp, so computing only the real frames could save
    most of this. That's an optimisation candidate for piece 2.
- **WASM: the encoder dominates, and it's fixed per call.** Measured here it's 7.3–8.5 s per call,
  whatever the segment length; scaled to your machine, an estimated 3.5–4 s. Decoding is about
  50–75 ms per token (scaled).
  - Estimated call cost ≈ **4 s + 0.2 s × segment length**. A final caption for a 10 s utterance
    would land about 6.5 s after it ends, with no queueing. That **misses the ≤ 6 s p50 WASM target**
    before any queueing or partials.
  - To keep up with continuous speech at all, a WASM segment has to be longer than about 5 s
    (cost < length). Back-to-back short utterances (2 s each) would fall behind, because each one
    pays the full encoder cost.
  - Consequences for piece 2:
    - **No partials on WASM.**
    - **Merge queued segments.** Finals that queue behind a busy worker are concatenated (up to
      12 s) into one call, paying the encoder cost once.
    - The **WASM target is likely unreachable with base q8 single-threaded.** See the decision
      needed below.

### Decision: the WASM fallback can't meet ≤ 6 s p50 as it stands

Options, in the order chosen: (1) multithreaded WASM via cross-origin isolation, (2) whisper-tiny on
WASM only, (3) relax the WASM target. Option 1 is built and measured below, and
[Chrome stable confirmed it](#chrome-stable-results-chrome-155): options 2 and 3 aren't needed.

### VAD worker (Silero v6.2.3 vs the energy fallback)

Measured with the bench's **VAD** panel: each fixture streamed at 48 kHz in 2048-sample chunks,
default tuning, 10 s maximum segment.

| Fixture | VAD    | Segments | Partials | Speech | ms / frame | Segments: start–end s (reason)                                                                                                                                      |
| ------- | ------ | -------: | -------: | -----: | ---------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| en      | silero |        7 |       22 |    86% |       1.02 | 0.32–5.70, 6.18–10.54, 11.01–17.90, 18.11–22.92, 23.49–32.87, 33.38–41.83 (silence); 42.40–47.71 (flush)                                                            |
| hi      | silero |        6 |       22 |    66% |       0.53 | 0.00–10.02 (max-length), 8.00–10.22, 17.63–27.65 (max-length), 25.63–30.76, 32.26–40.01, 42.40–49.99 (silence)                                                      |
| en      | energy |        8 |       27 |    85% |       0.01 | 0.32–5.61, 5.82–10.47 (silence), 10.56–20.58 (max-length), 18.56–22.86 (silence), 23.07–33.09, 31.07–41.09 (max-length), 39.07–41.80 (silence), 42.37–47.71 (flush) |
| hi      | energy |       10 |       21 |    55% |       0.01 | ten segments, several under 1.5 s                                                                                                                                   |

- **Silero follows the real pauses.** The English clip has six utterances; Silero cut seven
  segments at the gaps between them. It costs about 1 ms per 32 ms frame (3% of real time) on one
  WASM thread. It loads from the extension package, and no network request was made.
- **The energy fallback is usable but coarse.** It merges utterances into forced 10 s cuts on
  English and fragments the Hindi.
- **Forced cuts land badly.** The first Hindi sentence is 10.26 s, so the cut at exactly 10 s left
  a 2.2 s segment: 2 s of overlap plus 0.2 s of new speech, decoded again for almost nothing. When a
  cut is forced, the segmenter should cut at the quietest frame in the last 1–2 s, as Silero's
  `get_speech_timestamps` does. That's for the tuning piece.

## Cross-origin isolation (multithreaded WASM)

The manifest sets `cross_origin_embedder_policy: { value: 'require-corp' }` and
`cross_origin_opener_policy: { value: 'same-origin' }`, the MV3 keys from Chrome 93
([COEP](https://developer.chrome.com/docs/extensions/reference/manifest/cross-origin-embedder-policy),
[COOP](https://developer.chrome.com/docs/extensions/reference/manifest/cross-origin-opener-policy)).
`check:dist` fails if either changes.

Checked on Chromium 156:

| Context                                   | `crossOriginIsolated` | Notes                                               |
| ----------------------------------------- | --------------------- | --------------------------------------------------- |
| Popup, options page (e2e checks both)     | true                  |                                                     |
| Offscreen document (production and bench) | true                  |                                                     |
| Bench page and its ASR worker             | true                  | 4 threads: ORT starts 3 pthreads (4 worker targets) |
| Service worker                            | false                 | Doesn't matter: no ORT there                        |

- **Hugging Face downloads still work.** Model files are CORS fetches, and a CORS response
  satisfies `require-corp`. whisper-base q8 downloaded from `huggingface.co` and
  `us.aws.cdn.hf.co`, the Cache API stored it, and the next load came from cache in about 1 s. No
  other host was contacted.
- **ORT's pthreads need no CSP change.** ORT starts each one with
  `new Worker(new URL(import.meta.url), { type: 'module', name: 'em-pthread' })`. In our bundle
  that's the ASR worker's own chunk, a `'self'` URL with no `blob:`. Those copies skip the worker's
  message handler.
- **Tab capture broke, and is fixed.** With both headers set (either one alone is fine), a stream
  id from `tabCapture.getMediaStreamId` called in the service worker fails in the isolated
  offscreen document with `AbortError: Error starting tab capture`. An id requested in the popup,
  which is isolated like the offscreen document, works. So the popup now requests the id (it's the
  invoked context anyway) and sends it with `capture/start`. The P2 capture probe passes on the
  isolated build: capture, reopening the popup, mute, same- and cross-origin navigation, switching
  tabs, stop, and closing the tab.
- **The VAD worker is unchanged**: same segments, still single-threaded, still no network.

**Thread count.** By default, ORT gets half the cores, at most 4 (`defaultThreads` in
[threads.ts](../../src/asr/threads.ts)), so the tab being captioned keeps the rest. On this 12-thread
machine that's 4. ORT fixes the count when a worker first loads a model. The bench's **Threads**
selector restarts the worker and reloads the model from cache when it changes.

### Per-call cost: WASM q8, 1 vs 4 threads

Same session, bench window in front. This time the CPU wasn't slowed down: the 1-thread rows are
about half the earlier automated run, in line with the estimate for Chrome stable.

| Config                  | Segment (s) | Features (ms) | Encoder (ms) | Decoder (ms) | Decoder steps | ms / step | Total (ms) | Call RTF |
| ----------------------- | ----------: | ------------: | -----------: | -----------: | ------------: | --------: | ---------: | -------: |
| wasm q8 / q8, 1 thread  |           2 |           226 |         3827 |          666 |             7 |      98.8 |       4958 |    2.479 |
| wasm q8 / q8, 1 thread  |           5 |           168 |         3425 |         1322 |            18 |      68.5 |       5050 |    1.010 |
| wasm q8 / q8, 1 thread  |          10 |           284 |         4241 |         1606 |            26 |      67.1 |       6066 |    0.607 |
| wasm q8 / q8, 1 thread  |          15 |           219 |         4424 |         3080 |            49 |      69.0 |       7752 |    0.517 |
| wasm q8 / q8, 4 threads |           2 |           261 |         1358 |          589 |             7 |      75.8 |       2223 |    1.112 |
| wasm q8 / q8, 4 threads |           5 |           348 |         2461 |         1709 |            18 |      81.9 |       4265 |    0.853 |
| wasm q8 / q8, 4 threads |          10 |           345 |         2149 |         1803 |            26 |      73.9 |       4393 |    0.439 |
| wasm q8 / q8, 4 threads |          15 |           317 |         2013 |         3540 |            49 |      73.8 |       5853 |    0.390 |

The decoded text is identical at both thread counts.

- **The encoder roughly halves with 4 threads** (3.4–4.4 s → 1.4–2.5 s). Decoding doesn't speed up:
  each step is small and sequential, about 70–80 ms either way.
- **A 10 s utterance's final would land about 4.9 s after it ends** (the 0.5 s pause plus a 4.4 s
  call) with nothing queued. That's inside **≤ 6 s p50**, where 1 thread (about 6.6 s) misses. p95
  depends on queueing, which the live bench mode measures in piece 4.
- If Chrome stable confirms this, option 1 is enough and whisper-tiny isn't needed. The WASM
  plan stays as before: no partials, merge queued segments, and a maximum segment of about 10 s.

## Chrome stable results (Chrome 155)

Measured on Chrome 155 stable, bench window in front, whisper-base, before the feature cut below.
Totals per call:

| Config                  | 2 s    | 5 s    | 10 s   | 15 s   | Encoder      |
| ----------------------- | ------ | ------ | ------ | ------ | ------------ |
| wasm q8 / q8, 1 thread  | 3.2 s  | 3.9 s  | 4.1 s  | 5.1 s  | about 2.7 s  |
| wasm q8 / q8, 4 threads | 1.7 s  | 2.2 s  | 2.8 s  | 3.9 s  | 1.2–1.5 s    |
| webgpu fp16 / fp16      | 0.87 s | 1.67 s | 2.20 s | 2.71 s | about 0.32 s |

- Features took 130–250 ms on every call, whatever the segment length.
- The decoder took about 42–55 ms per step on both backends.
- VAD: English, 7 segments at natural pauses. Hindi, 6 segments, two of them cut at the 10 s
  maximum.
- Capture start through the popup works on Chrome stable: the meter moves, the tab stays audible,
  and Stop works.

**Decision: the WASM fallback is whisper-base q8 / q8 on 4 threads, and whisper-tiny isn't
needed.** A 10 s utterance's final lands about 3.3 s after it ends (the 0.5 s pause plus a 2.8 s
call), well inside ≤ 6 s p50.

## Why WASM q8 decoded fewer steps on 10 s segments

On Chrome stable, the median for the 10 s segments was 26 decoder steps on WASM q8 against 34 on
WebGPU fp16, while 5 s and 15 s matched closely. The three 10 s segments' texts show where the gap
comes from. Two match apart from a word or two. The middle one (starting at 18.85 s, mid-sentence)
differs:

- fp16: "He is drawn from eating and its results occur most readily to the mind. He has graved
  doubts whether Sir Frederick Layton's work is really Greek after all."
- q8: "He has graved doubts whether Sir Frederick Layton's work is really Greek after all."

q8 skips the clipped opening clause and transcribes the rest. It doesn't stop early: both decodes
end on "after all." and `<|endoftext|>`. It isn't the token cap either (224, against 19 tokens
here).

Which half is responsible? Same segment, WASM, Chromium:

| Encoder / decoder  | Middle segment            | Median steps |
| ------------------ | ------------------------- | -----------: |
| fp32 / fp32        | keeps "He is drawn from…" |           34 |
| q8 / fp32          | skips it                  |           26 |
| fp32 / q8          | skips it                  |           26 |
| q8 / q8            | skips it                  |           26 |
| webgpu fp16 / fp16 | keeps it                  |           34 |

Either quantized half is enough, because the choice is a near-tie. Every config decodes " He"
first. At the next token, fp32 gives " is" (carry on with the clipped sentence) log-prob −0.83 and
" has" (jump to the next sentence) −1.03: 44 % against 36 %. q8 noise reverses that order (q8 / q8:
−1.14 against −0.69). This was measured in Node with onnxruntime-node on the same segment, where
fp32 and fp16 decode the long version and every q8 mix decodes the short one, as in the browser.
In the q8 decoder, the uncached forward pass and the KV-cached decode can even disagree with each
other. That fits dynamic int8 quantization, which takes its activation scales from whatever it
processes at once: the whole sequence uncached, but one token at a time with the cache.

What it means for captions:

- It only bites on a segment that starts mid-sentence. The bench's fixed offsets do that; the VAD
  cuts at pauses, except for its forced 10 s cuts (Hindi had two).
- The quietest-point forced cut (tuning piece) moves those cuts to the quietest stretch, so they
  start at a word gap.
- Piece 2 prompts a decode with the previous segment's text, so Whisper continues mid-sentence
  instead of jumping to the next one. Conditioning on previous text is a known way for loops to
  propagate, so it's restricted:
  - only when the previous segment ended in a forced cut, not a natural pause;
  - only its last 20–30 tokens;
  - never when the previous segment failed a guard or was dropped as no-speech;
  - reset after a long silence or a language change.

  The bench measures WER and loop count with and without it.

**Decision: q8 / q8 stays the WASM fallback.** Its other texts match fp16's apart from a word or
two, and the skip above only happens on segments that start mid-sentence.

## Features for the real audio only

Whisper always sees 30 s. The feature extractor zero-pads shorter audio to 480 000 samples and
computes all 3000 STFT frames plus the mel projection, so a 5 s segment pays for 25 s of silence.
Every padded frame gives the same value, which can't raise the maximum used for normalisation. So
the worker now computes the frames up to just past the audio plus one all-zero frame, and repeats
that frame's value for the rest ([features.ts](../../src/asr/features.ts)). The output matches the
full extractor to within 5e-7 (a few float32 steps, from ONNX Runtime's matmul rounding a few
values differently with fewer columns). features.test.ts checks this on 13 clips, including empty,
silent, near-silent, 30 s and 40 s.

Feature time per call, Chromium, same session:

| Backend         | 2 s    | 5 s    | 10 s   | 15 s   |
| --------------- | ------ | ------ | ------ | ------ |
| webgpu, before  | 239 ms | 241 ms | 202 ms | 327 ms |
| webgpu, after   | 23 ms  | 51 ms  | 74 ms  | 98 ms  |
| wasm q8, before | 484 ms | 382 ms | 356 ms | 412 ms |
| wasm q8, after  | 34 ms  | 80 ms  | 125 ms | 182 ms |

The WASM "before" row ran alongside other work, so it's high. Chrome stable showed 130–250 ms.
WASM q8 texts are identical before and after. WebGPU fp16 changed one name's spelling in two
segments, but it does that from run to run anyway: two runs with the same features gave "Krilter"
and "Kralter".

## How to re-measure on Chrome stable

1. On `main`, run `npm ci` and `npm run build:bench`, then reload the `dist-bench` extension in
   `chrome://extensions`.
2. Open the bench page and **keep its window in front** while measuring. Under **Environment**,
   check that `crossOriginIsolated` is `true` and **Default ORT threads** shows `4 threads`.
3. Click **Clear per-call results**, select a model, device and dtypes, set **Threads**, click
   **Load model**, then **Measure per-call cost**. Repeat for each config.
4. Click **Copy results as Markdown**. The decoded texts aren't in the export; to compare them,
   open **Decoded text** under the per-call table.
