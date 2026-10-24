# Upstream issue draft: Whisper `generate()` ignores `encoder_outputs`

Status: **not filed yet** (to be filed by hand). Searched `huggingface/transformers.js` issues and
PRs on 2026-10-24 for `encoder_outputs`, `forward_params`, `_prepare_model_inputs`, "precomputed
encoder" and "encoder twice": no existing report. The only hit, #1739, is about a WebGPU memory leak.

Our workaround is in `src/asr/asr.worker.ts` (`allowPrecomputedEncoderOutput`).

---

## Title

Whisper: `generate({ encoder_outputs })` ignores the precomputed encoder output and runs the encoder again

## Versions

- `@huggingface/transformers` 4.3.1 (latest on npm, 2026-10-24). The same code is still on `main`
  at `94587c4`.
- Reproduced in Node 22.19.0 (onnxruntime-node 1.30.0, CPU) and in Chrome 155/156 on the WASM and
  WebGPU backends (onnxruntime-web).
- Model: `onnx-community/whisper-tiny.en` (repro), `onnx-community/whisper-base` (where we found it).

## Description

Passing a precomputed encoder output to `WhisperForConditionalGeneration.generate()` has no effect:
the encoder runs again inside `generate()`. Nothing errors, so the only symptom is that every call
pays for a second encoder pass. That's the most expensive part of a Whisper call on WASM.

Cause: `_prepare_model_inputs` keeps only the kwargs named in `this.forward_params`
([`modeling_utils.js#L708-L709`](https://github.com/huggingface/transformers.js/blob/94587c4652e3debc02f2e73ef688679d04792e3a/packages/transformers/src/models/modeling_utils.js#L708-L709)):

```js
const model_inputs = pick(model_kwargs, this.forward_params);
```

and `generate()` skips the encoder only when `encoder_outputs` survived that pick
([`#L871`](https://github.com/huggingface/transformers.js/blob/94587c4652e3debc02f2e73ef688679d04792e3a/packages/transformers/src/models/modeling_utils.js#L871)):

```js
} else if (!('encoder_outputs' in model_inputs)) {
    model_inputs = await this._prepare_encoder_decoder_kwargs_for_generation({ ... });
```

`WhisperPreTrainedModel.forward_params` doesn't list `encoder_outputs`
([`modeling_whisper.js#L19`](https://github.com/huggingface/transformers.js/blob/94587c4652e3debc02f2e73ef688679d04792e3a/packages/transformers/src/models/whisper/modeling_whisper.js#L19)):

```js
forward_params = [
  'input_features',
  'attention_mask',
  'decoder_input_ids',
  'decoder_attention_mask',
  'past_key_values',
];
```

Other encoder-decoder models do list it (`T5PreTrainedModel`, `MusicgenForConditionalGeneration`,
Florence2), so for them the same call reuses the encoder output.

## Reproduction

```sh
npm i @huggingface/transformers@4.3.1
node repro.mjs
```

```js
// repro.mjs: counts calls to the encoder session (`sessions.model`) inside generate().
import { AutoProcessor, Tensor, WhisperForConditionalGeneration } from '@huggingface/transformers';

const MODEL = 'onnx-community/whisper-tiny.en';
const processor = await AutoProcessor.from_pretrained(MODEL);
const model = await WhisperForConditionalGeneration.from_pretrained(MODEL, { dtype: 'fp32' });

const encoder = model.sessions.model;
const run = encoder.run.bind(encoder);
let encoderRuns = 0;
encoder.run = (...args) => {
  encoderRuns += 1;
  return run(...args);
};

// 5 s of a 440 Hz tone; the content doesn't matter.
const audio = Float32Array.from(
  { length: 16_000 * 5 },
  (_, i) => 0.1 * Math.sin((2 * Math.PI * 440 * i) / 16_000),
);
const { input_features } = await processor(audio);

// Run the encoder once ourselves (e.g. to reuse its output for a no-speech check).
const { last_hidden_state } = await encoder.run({ input_features: input_features.ort_tensor });
const encoder_outputs = new Tensor(last_hidden_state);

async function generateWithEncoderOutputs(label) {
  encoderRuns = 0;
  const started = performance.now();
  const tokens = await model.generate({
    inputs: input_features,
    encoder_outputs,
    max_new_tokens: 8,
  });
  const ms = (performance.now() - started).toFixed(0);
  console.log(`${label}: encoder ran ${encoderRuns} time(s) inside generate() (${ms} ms)`);
  return tokens.tolist().flat().join(',');
}

const shipped = await generateWithEncoderOutputs('as shipped');
// Workaround: let `encoder_outputs` through _prepare_model_inputs' pick(kwargs, forward_params).
model.forward_params.push('encoder_outputs');
const patched = await generateWithEncoderOutputs('with encoder_outputs in forward_params');
console.log(`same tokens: ${String(shipped === patched)}`);
```

Output (Windows 11, Node 22.19.0, CPU):

```text
as shipped: encoder ran 1 time(s) inside generate() (302 ms)
with encoder_outputs in forward_params: encoder ran 0 time(s) inside generate() (168 ms)
same tokens: true
```

## Expected vs actual

- **Expected:** with `encoder_outputs` passed, `generate()` uses it and runs only the decoder, as it
  already does for T5 and Musicgen.
- **Actual:** the argument is silently dropped, and the encoder runs again on `inputs`. The output
  is the same, but each call does an extra encoder pass.

## Impact

Reusing one encoder pass is how you do a cheap no-speech check (one decoder step at
`<|startoftranscript|>`), language detection, or temperature-fallback retries before or around the
real decode. With this bug, each of those costs a full encoder pass.

Measured in Chrome with `onnx-community/whisper-base`, one call on a 2–15 s segment:

| Backend                        | Encoder pass | Extra time per call from the bug |
| ------------------------------ | -----------: | -------------------------------: |
| WASM q8, single thread         |   7.3–8.5 s¹ |                     about 4.5 s¹ |
| WebGPU fp16                    |  0.32–0.39 s |                     about 0.35 s |
| Node CPU, tiny.en fp32 (repro) |            — |  134 ms of a 302 ms `generate()` |

¹ Measured in Chromium under automation, which ran CPU work about 2× slower than a normal Chrome
window on the same machine.

## Suggested fix

Add `'encoder_outputs'` to `WhisperPreTrainedModel.forward_params`, as T5 and Musicgen do. The
decoder path already handles it: `seq2seq_forward` uses `encoder_outputs` when present and only
runs the encoder otherwise. Alternatively, `_prepare_model_inputs` could always keep
`encoder_outputs` for encoder-decoder models, which would cover any other models missing it.
