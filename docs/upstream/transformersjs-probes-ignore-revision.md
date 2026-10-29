# Upstream draft: a pinned `pipeline()` load still requests `resolve/main/...`

Status: **not posted yet** (to be posted by hand). This is already reported, and fixed in an open
PR: [huggingface/transformers.js#1751](https://github.com/huggingface/transformers.js/pull/1751),
"Pass `revision` through the file-listing and config paths". It's approved, mergeable and unmerged;
the author last rebased it on 2026-10-03, and another user reported the Whisper ASR case there on
4.3.0. Searched issues and PRs on 2026-10-29 for "revision get_file_metadata", "revision
resolve/main", "get_pipeline_files revision", "tokenizer_config revision" and "pinned revision":
#1751 is the only match.

So **don't file a new issue**. Post the comment below on #1751: it adds a second minimal repro, the
full list of requests a pinned ASR load sends to `main` on 4.3.1, and a concrete breakage (a
cache-only load that fails). If #1751 is closed without merging, the full issue text further down
stands on its own.

Our workaround is in `src/asr/models.ts` (`pinnedUrl`) and the worker's `env.fetch` in
`src/asr/asr.worker.ts`. They rewrite `/resolve/main/` to the pinned revision and answer from the
cache first. Once a release includes #1751, check whether the rewrite is still needed.

---

## Comment for #1751

> Another repro on 4.3.1 (Node 24.22.1), for the ASR pipeline. It counts every request by the revision it
> names, from a cold cache:
>
> ```js
> // probes.mjs: lists every request a pinned pipeline() load makes, grouped by the revision it names.
> import { mkdtempSync } from 'node:fs';
> import { tmpdir } from 'node:os';
> import { join } from 'node:path';
> import { env, pipeline } from '@huggingface/transformers';
>
> const MODEL = 'onnx-community/whisper-tiny.en';
> const REVISION = '2575352d61be1bf7225cf8f8b268a4678025fc58';
>
> env.cacheDir = mkdtempSync(join(tmpdir(), 'tfjs-probes-')); // a cold cache, so every file is fetched
>
> const requests = [];
> const fetchFile = env.fetch;
> env.fetch = (url, init) => {
>   const range = new Headers(init?.headers).get('range');
>   requests.push(`${String(url).split('/resolve/')[1]}${range ? `  (Range ${range})` : ''}`);
>   return fetchFile(url, init);
> };
>
> await pipeline('automatic-speech-recognition', MODEL, {
>   revision: REVISION,
>   dtype: 'q8',
>   progress_callback: () => {}, // pipeline() only gathers progress totals when one is given
> });
>
> const atMain = requests.filter((r) => r.startsWith('main/'));
> const pinned = requests.filter((r) => r.startsWith(`${REVISION}/`));
> console.log(`pinned to ${REVISION.slice(0, 7)}: ${pinned.length} requests`);
> for (const r of pinned) console.log(`  ${r.slice(REVISION.length + 1)}`);
> console.log(`at main: ${atMain.length} requests`);
> for (const r of atMain) console.log(`  ${r.slice('main/'.length)}`);
> ```
>
> ```text
> pinned to 2575352: 11 requests
>   config.json
>   config.json  (Range bytes=0-0)
>   preprocessor_config.json
>   tokenizer.json
>   tokenizer_config.json
>   onnx/encoder_model_quantized.onnx
>   onnx/decoder_model_merged_quantized.onnx
>   generation_config.json
>   tokenizer.json  (Range bytes=0-0)
>   tokenizer_config.json  (Range bytes=0-0)
>   generation_config.json  (Range bytes=0-0)
> at main: 7 requests
>   config.json
>   tokenizer_config.json  (Range bytes=0-0)
>   preprocessor_config.json  (Range bytes=0-0)
>   onnx/encoder_model_quantized.onnx  (Range bytes=0-0)
>   onnx/decoder_model_merged_quantized.onnx  (Range bytes=0-0)
>   generation_config.json  (Range bytes=0-0)
>   tokenizer.json  (Range bytes=0-0)
> ```
>
> Each of the seven goes through a path this PR fixes:
>
> - `config.json` comes from `get_model_files` → `get_config(modelId, { config })`;
> - `tokenizer_config.json` and `preprocessor_config.json` come from `get_tokenizer_files` /
>   `get_processor_files`, which pass `{}`;
> - the rest come from the progress-totals pass in `pipeline()`, `get_file_metadata(model, file)`.
>
> Where it bites us: a browser extension that ships with a fixed model revision, caches the files
> after the user agrees to a download, and refuses network requests otherwise (a custom `env.fetch`).
> With every pinned file cached, the next load still fails, because it asks for
> `resolve/main/tokenizer_config.json`, which was never cached. We now rewrite `/resolve/main/` to the
> pinned revision in `env.fetch` and answer from the cache first. With this PR, that wouldn't be needed.

---

## Full issue text (only if #1751 is closed without merging)

### Title

`pipeline()` with a pinned `revision` still requests `config.json` and file probes from `resolve/main`

### Versions

- `@huggingface/transformers` 4.3.1 (latest on npm, 2026-10-29). The same calls are on `main` at
  `656d627`.
- Node 24.22.1 (onnxruntime-node, CPU), and Chrome (onnxruntime-web) in a browser extension.
- Model: `onnx-community/whisper-tiny.en` (repro), `onnx-community/whisper-base` (where we found
  it).

### Description

`pipeline(task, model, { revision })` loads the weights from `revision`, but the steps that decide
which files to load, and how big they are, read from `main`:

- `pipeline()` passes only `{ device, dtype }` to `get_pipeline_files()`
  ([`pipelines.js#L137`](https://github.com/huggingface/transformers.js/blob/656d627317100ac9f9bb7aa7952297b71f8c37b5/packages/transformers/src/pipelines.js#L137)),
  and its progress-totals pass calls `get_file_metadata(model, file)` with no options
  ([`#L147`](https://github.com/huggingface/transformers.js/blob/656d627317100ac9f9bb7aa7952297b71f8c37b5/packages/transformers/src/pipelines.js#L147)).
- `get_files()` calls `get_model_files`, `get_tokenizer_files` and `get_processor_files` without the
  revision
  ([`get_files.js#L30-L37`](https://github.com/huggingface/transformers.js/blob/656d627317100ac9f9bb7aa7952297b71f8c37b5/packages/transformers/src/utils/model_registry/get_files.js#L30-L37)).
- `get_tokenizer_files()` and `get_processor_files()` take no options and pass `{}` to
  `get_file_metadata()`
  ([`get_tokenizer_files.js#L15`](https://github.com/huggingface/transformers.js/blob/656d627317100ac9f9bb7aa7952297b71f8c37b5/packages/transformers/src/utils/model_registry/get_tokenizer_files.js#L15),
  [`get_processor_files.js#L17`](https://github.com/huggingface/transformers.js/blob/656d627317100ac9f9bb7aa7952297b71f8c37b5/packages/transformers/src/utils/model_registry/get_processor_files.js#L17)).

### Reproduction

The script and its output are in the comment above: 7 of a pinned load's 18 requests name `main`,
including a full GET of `config.json`.

### Expected vs actual

- **Expected:** every request names the pinned revision.
- **Actual:** the file list comes from `main`'s `config.json`, and the existence and size probes ask
  `main`. If `main` has moved, the listing can name files that don't exist at the pinned commit. A
  load restricted to cached files fails, because the `main` probes were never cached.

### Suggested fix

Forward the pretrained options (`revision`, `cache_dir`, `local_files_only`) through
`get_pipeline_files` → `get_files` → `get_model_files` / `get_tokenizer_files` /
`get_processor_files`, and into the progress-totals `get_file_metadata` calls, as #1751 does.
