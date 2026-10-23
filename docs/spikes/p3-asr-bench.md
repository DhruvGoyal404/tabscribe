# P3a: on-device ASR research and benchmark

**Goal:** choose the Whisper model, device and dtype for live captions from measurements on the
target machine (Windows), not from library defaults. P3a builds the ASR worker, the resampler and a
benchmark. Live captions are P3b.

**Status:** done. Measured on the target machine (see [Results](#results)).

**Decision:** default to **whisper-base on WebGPU, fp16/fp16**. Fall back to **whisper-base on WASM,
q8/q8** when there's no hardware adapter. **English-first:** Hindi isn't shippable on-device yet
(see [Decisions](#decisions)).

## Versions and pins

| What                           | Pin                                        | Notes                                                                                                             |
| ------------------------------ | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `@huggingface/transformers`    | `4.3.1`                                    | Newest release past the 14-day rule (published 2026-10-06; 4.2.0 was 2026-04-22). npm audit clean.                |
| `onnxruntime-web`              | `1.31.0-dev.20260914-8d85527a0`            | Pinned exactly by Transformers.js. It's an ORT nightly, not a stable release.                                     |
| Node-only deps                 | `onnxruntime-node` 1.30.0, `sharp` 0.35.5  | Pulled in by Transformers.js but never bundled. npm 11 blocks their install scripts because they aren't approved. |
| `onnx-community/whisper-tiny`  | `ff4177021cc41f7db950912b73ea4fdf7d01d8e7` | Multilingual. The repos declare no licence; the weights are converted from `openai/whisper-*` (Apache-2.0).       |
| `onnx-community/whisper-base`  | `1846881b6b3a3024392c1eea3ad983695bc23925` |                                                                                                                   |
| `onnx-community/whisper-small` | `36050c46d777d46dc4b5f43f6d90574fc38f8732` |                                                                                                                   |

### File sizes per dtype (MB, at the pinned revisions)

Transformers.js loads `onnx/encoder_model{suffix}.onnx` and `onnx/decoder_model_merged{suffix}.onnx`,
plus about 2.8 MB of JSON (config, generation config, preprocessor and tokenizer).

| dtype (suffix)    | tiny enc | tiny dec | base enc | base dec | small enc | small dec |
| ----------------- | -------: | -------: | -------: | -------: | --------: | --------: |
| fp32 (none)       |     32.9 |    118.6 |     82.5 |    208.5 |     352.8 |     615.3 |
| fp16 (`_fp16`)    |     16.5 |     59.6 |     41.3 |    104.7 |     176.6 |     308.6 |
| q8 (`_quantized`) |     10.1 |     30.7 |     23.2 |     53.7 |      92.3 |     156.8 |
| int8 (`_int8`)    |     10.1 |     30.7 |     23.2 |     53.7 |      92.3 |     156.8 |
| uint8 (`_uint8`)  |     10.1 |     30.7 |     23.2 |     53.7 |      92.3 |     156.8 |
| q4 (`_q4`)        |      9.0 |     86.7 |     18.8 |    123.6 |      66.2 |     233.1 |
| bnb4 (`_bnb4`)    |      8.6 |     86.1 |     17.6 |    122.0 |      60.9 |     226.1 |

The 4-bit _decoders_ are bigger than the 8-bit ones. Only their MatMuls are 4-bit; the large token
embedding is presumably left in fp32. So "q4" doesn't mean "smallest".

## Step 0 findings

### WebGPU in a worker spawned by the offscreen document: available

Verified in Chromium 156 (headed) on the target machine. The service worker created an offscreen
document (reason `WORKERS`), the document spawned the ASR worker, and the worker called
`navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })`:

| Field                       | Value                               |
| --------------------------- | ----------------------------------- |
| vendor / architecture       | intel / gen-12lp                    |
| isFallbackAdapter           | false                               |
| shader-f16                  | yes                                 |
| maxBufferSize               | 2147.5 MB                           |
| maxStorageBufferBindingSize | 2147.5 MB                           |
| crossOriginIsolated         | false, so WASM runs single-threaded |

The bench page shows the same adapter line under **Environment**. Please confirm it on Chrome
stable.

### Model download hosts and CSP

Traced with redirects followed, and confirmed by the requests the browser made during a real load:

- `https://huggingface.co/<repo>/resolve/<sha>/<file>`: JSON files 307 to the same-origin
  `/api/resolve-cache/…`.
- ONNX weights 302 to the Xet CDN, `https://us.aws.cdn.hf.co/xet-bridge-us/…` from here. The
  hostname depends on region and storage backend.
- Both reply with `Access-Control-Allow-Origin` (`*` or the extension origin), so fetch works from
  extension pages and workers **without `host_permissions`**.

**CSP (applied):** `connect-src 'self' https://huggingface.co https://*.hf.co`, added to the
existing `script-src 'self' 'wasm-unsafe-eval'; object-src 'self'`. `*.hf.co` is Hugging Face's
CDN domain. It's needed because the CDN hostname isn't fixed, and it covers any depth
(`us.aws.cdn.hf.co`). Verified: `fetch('https://example.com')` from the bench page is refused by
CSP, and a full model load contacted only `huggingface.co` and `us.aws.cdn.hf.co`. CI's `check:dist`
fails if the CSP or the permissions change.

### Gotchas found in the library (verified in the 4.3.1 source)

1. **ORT's wasm comes from jsDelivr by default.** On import, Transformers.js sets
   `wasmPaths` to `cdn.jsdelivr.net/npm/onnxruntime-web@…`, which MV3 forbids. The worker
   overrides it with the bundled `ort-wasm-simd-threaded.asyncify.wasm` (26.9 MB, the build ORT
   1.31 uses for WebGPU) before any session is created.
2. **`env.useWasmCache` must be false.** It turns ORT's JS loader into a `blob:` URL, which
   `script-src 'self'` blocks. The file is local to the extension anyway.
3. **There is no language auto-detection.** The pipeline's type docs say `language: null` means
   auto-detect, but the code logs "No language specified - defaulting to English" and decodes as
   English. The worker detects the language itself: one decoder step from
   `<|startoftranscript|>` on the first 30 s, then argmax over the language tokens. That's the same
   approach as openai/whisper's `detect_language`.
4. Model files are cached with the Cache API (`transformers-cache`), keyed by their pinned
   `resolve/<sha>/…` URL. Clearing site data or the bench's **Clear model cache** button removes
   them.

## What P3a adds

- **`src/asr/asr.worker.ts`:** a dedicated worker that loads a Whisper pipeline with an explicit
  `{model, revision, device, dtype}`. Encoder and decoder dtypes are chosen separately. The worker
  transcribes (language forced or auto-detected) and translates to English. Requests name a
  catalogue key, never a repo or revision. `env.allowLocalModels = false`.
- **`src/asr/models.ts`:** the pinned catalogue, with exact per-file sizes.
- **`src/asr/protocol.ts`:** page ↔ worker messages with hand-written guards.
- **`src/asr/resample.ts`:** a streaming Kaiser-windowed sinc resampler from the native rate to
  16 kHz mono. -6 dB at 7.2 kHz, about 80 dB stopband, exact integer phase tracking, and output that
  doesn't depend on chunk size. P3b feeds it from the worklet.
- **`src/bench/`:** the dev-only bench page, fixtures and WER/CER. It's built only by
  `npm run build:bench` into `dist-bench/`, and CI checks it never lands in `dist/`.

## Method

- **Same code path as production.** The bench page spawns `src/asr/asr.worker.ts` exactly as the
  offscreen document will. Each fixture is decoded at **48 kHz**, which is what tab capture
  delivers, and run through our resampler to 16 kHz before inference.
- **Fixtures** ([README](../../src/bench/fixtures/README.md)):
  - 47.7 s of English read speech (LibriSpeech dev-clean, CC BY 4.0).
  - 50.7 s of Hindi read speech (FLEURS hi_in dev, CC BY 4.0, five sentences), with the matching
    FLoRes English sentences as the translation reference.
- **Each "Run benchmark"** makes six calls on the loaded model, in order:
  1. English, forced `en`: first run, which includes WebGPU shader compilation and warm-up.
  2. English, forced `en`: second run. Its transcript is scored.
  3. English, auto-detect.
  4. Hindi, forced `hi`: scored.
  5. Hindi, auto-detect.
  6. Hindi → English translate. The raw output is shown for judging by hand.
- **Long audio** uses the pipeline's chunking (`chunk_length_s: 30`, `stride_length_s: 5`), greedy
  decoding, no timestamps.
- **Metrics:**
  - **Load (s):** time for `pipeline()` to resolve: download (or cache read) plus session creation.
    The **Load** column says whether every file was already cached beforehand.
  - **RTF:** processing time ÷ audio duration. Below 1 is faster than real time. Live captions need
    well under 1, because P3b also has VAD, chunk overlap and UI work.
  - **WER / CER:** Levenshtein over normalised text ([metrics.ts](../../src/bench/metrics.ts)).
    Normalisation: NFKC, lower case, `[…]` annotations dropped, punctuation removed (Devanagari
    vowel signs kept), Devanagari digits mapped to ASCII. English also gets numbers spelled out and
    one spelling for titles ("Mr." = "MISTER"). Hindi CER counts code points with spaces removed.
  - **Detect:** the language picked by auto-detect, and the time detection took (the average of
    the two runs).
  - **Peak memory:** not measurable from the page. `performance.measureUserAgentSpecificMemory`
    needs cross-origin isolation, and the worker's WASM and GPU memory aren't in the JS heap. Read it
    from Chrome's Task Manager instead (see below).

## How to run it

1. Build:
   ```sh
   git checkout feat/p3a-asr-bench && npm ci
   npm run build:bench          # writes dist-bench/ (the production dist/ is unaffected)
   ```
2. Load it: `chrome://extensions` → Developer mode on → **Load unpacked** → choose `dist-bench/`.
   It's a separate unpacked extension from your normal `dist/` one; disable that one while
   benchmarking to avoid confusion.
3. Open the bench page. On the dist-bench card in `chrome://extensions`, copy the **ID**, then open
   `chrome-extension://<ID>/src/bench/index.html` in a tab.
4. Check **Environment**: the "WebGPU in worker" line should name your GPU and say "hardware
   adapter". That's Step 0 on Chrome stable.
5. For each configuration in the matrix below:
   1. Pick **Model**, **Device**, **Encoder dtype** and **Decoder dtype**. The line under them shows
      the exact download size and how much is already cached.
   2. Open Chrome's Task Manager (**Shift+Esc**) and sort by Memory footprint.
   3. Click **Load model (downloads … MB)**. This is the first-run load.
   4. Click **Run benchmark** and wait for "Done" (whisper-small on WASM can take minutes). Watch
      the peak **Memory footprint** of the bench tab's row and the **GPU Process** row, and note
      both.
   5. **Cached load:** press **Ctrl+R** to reload the page (a fresh worker; results are kept), pick
      the same configuration, click **Load model (from cache)**, then **Run benchmark** again. The
      new row's Load column says `cached`.
6. When done, click **Copy results as Markdown** and paste it back, together with:
   - the peak memory per configuration (tab row and GPU Process row);
   - a 1–5 adequacy score for each Hindi → English translation (expand **Outputs**; the FLoRes
     reference is under each one);
   - anything odd, such as `[Music]`-only output, repetition loops, or a configuration that failed
     to load (the error appears in red).

**Clear model cache** deletes every downloaded file. Use it only to re-measure a first-run download.

### Suggested matrix (in priority order)

| #   | Model | Device | Enc / Dec   | Download | Why                                                                   |
| --- | ----- | ------ | ----------- | -------: | --------------------------------------------------------------------- |
| 1   | base  | webgpu | fp32 / fp32 | 293.8 MB | Reference quality on GPU                                              |
| 2   | base  | webgpu | fp32 / q4   | 208.8 MB | The common WebGPU recipe; checks the reported Windows `[Music]` issue |
| 3   | base  | webgpu | fp16 / fp16 | 148.8 MB | Uses shader-f16; half the download of #1                              |
| 4   | base  | wasm   | q8 / q8     |  79.7 MB | The WASM fallback as Transformers.js would default it                 |
| 5   | tiny  | wasm   | fp32 / fp32 | 154.2 MB | The planned WASM fallback model                                       |
| 6   | small | webgpu | fp16 / fp16 | 488.0 MB | Is small's Hindi worth 3× the download?                               |
| 7   | small | webgpu | fp32 / q4   | 588.7 MB | Only if #6 misbehaves                                                 |

Rows 1–5 download about 0.8 GB in total (shared files are cached once). With #6 it's about 1.3 GB.

## Smoke test (whisper-tiny only, Chromium 156, this machine)

Run to validate the harness, not as results:

- **The pipeline is correct.** whisper-tiny fp32/fp32 on WASM, and fp32/q4 on WebGPU (Intel
  gen-12lp), both transcribed the English clip almost perfectly ("similes drawn" →
  "similezed on", "Leighton's" → "Latins'"). Auto-detect returned `en` and `hi` correctly. No
  `[Music]` problem with a q4 decoder on this Intel GPU, but the reported issue may be specific to
  other GPUs or to whisper-base. Rows 2 and 6 will show it.
- **The q8 decoder broke whisper-tiny.** q8/q8 on WASM looped ("the master of the master of…") on
  the first English window and on all the Hindi. The fp32 decoder didn't. So the WASM default
  (`q8`) can't be trusted without measuring; row 4 checks this for base. _Update: in the real runs,
  whisper-base q8/q8 on WASM never looped on English (4.4% WER). The loop was specific to tiny's
  q8 decoder._
- **whisper-tiny's Hindi is unusable** in every configuration tried (repetition loops). That's
  consistent with tiny's known weak Hindi. base and small are the candidates.
- Greedy decoding has no protection against repetition loops. P3b should add Whisper's usual
  guards: a compression-ratio or `no_repeat_ngram_size` check, and re-decoding at higher
  temperature on failure.

## Results

Run by hand with the bench on **Chrome 155 stable**, Windows, Intel gen-12lp (WebGPU hardware
adapter, shader-f16), 12 CPU threads, not cross-origin isolated (single-threaded WASM).

| #   | Model | Device | Enc / Dec   | Load (s) | EN RTF 1st | EN RTF 2nd | EN WER | HI RTF | HI CER | HI WER | Detect EN / HI | Detect (ms) |
| --- | ----- | ------ | ----------- | -------: | ---------: | ---------: | -----: | -----: | -----: | -----: | -------------- | ----------: |
| 1   | base  | webgpu | fp32 / fp32 |    245.6 |      0.505 |      0.327 |   5.3% |  0.451 |  81.4% |  92.6% | en / hi        |        4380 |
| 2   | base  | webgpu | fp32 / q4   |     90.6 |      0.403 |      0.309 |   4.4% |  1.019 | 135.0% |  94.4% | en / hi        |        3988 |
| 3   | base  | webgpu | fp16 / fp16 |     95.5 |      0.290 |      0.188 |   5.3% |  0.215 |  80.7% |  91.7% | en / hi        |         692 |
| 4   | base  | wasm   | q8 / q8     |     51.3 |      0.249 |      0.266 |   4.4% |  1.019 |  90.1% | 168.5% | en / hi        |        3189 |
| 5   | tiny  | wasm   | fp32 / fp32 |    103.4 |      0.113 |      0.137 |   3.5% |  0.437 | 142.8% | 221.3% | en / hi        |        1772 |
| 6   | small | webgpu | fp16 / fp16 |    323.5 |      1.042 |      0.781 |   2.7% |  1.659 |  64.3% | 134.3% | en / hi        |        2281 |

Load times scale with download size, so they're dominated by the first-run download over the
tester's connection. Cached-load times weren't captured. Hindi CER/WER above 100% means the output
was longer than the reference (repetition loops); see below.

**Peak memory** (Chrome Task Manager):

- Bench tab ("Extension: Tabscribe ASR bench"): about 1.3–1.44 GB on the whisper-base WebGPU rows
  (1–3).
- GPU Process: about 1.5 GB on row 2.
- Not captured for the other rows.

**Hindi → English translation** (adequacy 1–5, judged against the FLoRes reference):

| #   | Config                   | Adequacy | Notes                                                                                                              |
| --- | ------------------------ | -------: | ------------------------------------------------------------------------------------------------------------------ |
| 1   | base webgpu fp32 / fp32  |        1 | Repetition loops                                                                                                   |
| 2   | base webgpu fp32 / q4    |        1 | Repetition loops                                                                                                   |
| 3   | base webgpu fp16 / fp16  |        1 | Repetition loops                                                                                                   |
| 4   | base wasm q8 / q8        |        1 | Repetition loops                                                                                                   |
| 5   | tiny wasm fp32 / fp32    |        1 | Repetition loops                                                                                                   |
| 6   | small webgpu fp16 / fp16 |        2 | No loop in the translation, and the first sentence is partly right. The Hindi transcript still loops ("ये ये ये"). |

Examples of the loops in rows 1–5: "the kabut of the kabut", "a kind of a kind of", "the people of
the people", "chach ke parisar on the paper".

**Other observations:**

- No `[Music]`-only output in any row, including the q4 decoder on WebGPU (row 2). The reported
  Windows issue doesn't reproduce on this Intel GPU. It may still appear on other GPU vendors.
- English never looped in any row, including base q8/q8 on WASM.
- fp16/fp16 (row 3) is the fastest WebGPU configuration on this GPU (EN RTF 0.188 warm), with half
  the download of fp32/fp32. Its language detection is also much faster (692 ms, against 3–4 s
  elsewhere).
- First-run RTF is higher than second-run RTF on WebGPU (shader compilation and warm-up). P3b
  should warm the model up once after loading, before live audio arrives.

### Caveats

- **The fixtures are single, short, clean clips:** 47.7 s of read English from one speaker, and
  50.7 s of read Hindi. On about 105 reference words, one word is about 1% WER. So the English
  spread (2.7–5.3%) is within noise, and doesn't rank the configurations. The clips are good enough
  to catch breakage (loops, `[Music]`, wrong language) and to compare speed, not to compare
  accuracy. P3b adds a longer, realistic English regression fixture.
- RTF is from one run per configuration on one machine. Treat differences under about 20% as noise.
- **ONNX Runtime is a nightly.** Transformers.js 4.3.1 pins `onnxruntime-web`
  `1.31.0-dev.20260914-8d85527a0`, a dev build rather than a stable ORT release. A future
  Transformers.js or ORT bump can change numerics or WebGPU kernels (for example the reported q4
  overflow), so any bump must re-run this bench, at least rows 3 and 4. The exact pins and the
  14-day rule keep it from changing silently.

## Decisions

1. **Default: whisper-base, WebGPU, fp16 / fp16.** It's the fastest WebGPU configuration measured
   (warm EN RTF 0.188, HI RTF 0.215), a 148.8 MB download, and has the fastest language detection.
   Its English accuracy matches fp32/fp32 within noise.
2. **Fallback: whisper-base, WASM, q8 / q8**, when there's no WebGPU adapter or only a fallback
   (software) adapter. EN RTF 0.25–0.27 single-threaded, a 79.7 MB download, and no English loops.
3. **Not tiny.** Its better WER (3.5%) is within noise on one 48 s clip, and keeping one model means
   one download and one set of behaviours to test.
4. **whisper-small rejected for live captions.** EN RTF 0.78–1.04 leaves no headroom for real time
   on this machine, and the download is 488 MB.
5. **English-first: Hindi isn't shippable on-device yet.** Every configuration loops on Hindi
   transcription, and translation adequacy is 1–2 out of 5. P3b ships English. The README makes no
   Hindi or translation promise until it passes this bench. Hindi gets its own spike (P3c).

## Next

- **P3b (live English captions):** streaming pipeline with repetition guards, built on the
  decisions above. Plan agreed before coding.
- **P3c (Hindi spike, after P3b):**
  - Re-run Hindi with the P3b repetition guards on base fp16 and small fp16.
  - Survey openly licensed Hindi fine-tuned Whisper models (base/small) that have ONNX weights or
    can be converted. Report licence, size and revision.
  - Plan B: Hindi through the P6 opt-in bring-your-own-key cloud mode.
- **Multithreaded WASM:** extension pages aren't cross-origin isolated, so the WASM fallback runs
  single-threaded. Setting `cross_origin_embedder_policy` / `cross_origin_opener_policy` could
  enable threads. Evaluate it against our CORS model downloads before relying on it.
