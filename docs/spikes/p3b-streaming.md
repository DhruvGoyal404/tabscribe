# P3b: streaming English captions

Live English captions from tab audio, built on the P3a decisions (whisper-base: WebGPU fp16/fp16,
fallback WASM q8/q8). Built in PR-sized pieces. This doc grows with each piece.

| Piece | Scope                                                                                      | Status   |
| ----- | ------------------------------------------------------------------------------------------ | -------- |
| 1     | Per-call cost measurement (encoder vs decoder) and the VAD worker                          | **this** |
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

### Decision needed: the WASM fallback can't meet ≤ 6 s p50 as it stands

Options, to decide once the WASM rows are re-measured on Chrome stable:

1. **Multithreaded WASM.** Cross-origin isolation (`cross_origin_embedder_policy` +
   `cross_origin_opener_policy` in the manifest) enables ORT's threads. The encoder is mostly matrix
   multiplication and should scale with the 12 threads. This needs checking against our CORS model
   downloads and the offscreen document.
2. **whisper-tiny on WASM only.** Its encoder is several times cheaper (your bench: EN RTF 0.113 vs
   base q8 0.266). It's a second download and model to test, but only on machines without WebGPU.
3. **Relax the WASM target**, for example to ≤ 10 s p50 / ≤ 16 s p95, and say so in the UI.

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

## How to re-measure on Chrome stable

1. `npm run build:bench`, then reload the `dist-bench` extension in `chrome://extensions`.
2. Open the bench page and **keep its window in front** while measuring.
3. **Per-call cost:** load whisper-base **wasm q8/q8**, click **Measure per-call cost**, and wait
   (about a minute). Then do the same for **webgpu fp16/fp16**.
4. **VAD:** click **Run VAD on fixtures** (no download).
5. Click **Copy results as Markdown** and paste it back. The export includes both new tables.
