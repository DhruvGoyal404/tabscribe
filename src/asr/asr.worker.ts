// ASR worker: loads a pinned Whisper model with Transformers.js and transcribes 16 kHz mono audio.
// It runs in a dedicated worker so inference never blocks the page that spawned it (the bench page
// now, the offscreen document in P3b).
//
// Network and code-loading rules (MV3 forbids remote code):
// - Model files come only from huggingface.co at the revision pinned in models.ts. Requests name a
//   catalogue key, never a repo or revision, and local model paths are disabled.
// - The ONNX Runtime WebAssembly binary is bundled with the extension. Transformers.js would
//   otherwise point ORT at jsDelivr, so wasmPaths is set before any session is created.
// - Model files are cached with the Cache API (`transformers-cache`), keyed by their pinned URL.
import {
  env,
  pipeline,
  Tensor,
  type AutomaticSpeechRecognitionPipeline,
  type ProgressInfo,
} from '@huggingface/transformers';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import { HF_HOST, MODELS } from './models';
import {
  isWorkerRequest,
  type AsrTask,
  type GpuInfo,
  type WorkerRequest,
  type WorkerResponse,
} from './protocol';
import { ASR_SAMPLE_RATE } from './resample';

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.remoteHost = HF_HOST;
env.remotePathTemplate = '{model}/resolve/{revision}/';
env.useBrowserCache = true;
// The wasm cache would turn ORT's loader into a blob: URL, which `script-src 'self'` blocks. The
// files are local to the extension anyway.
env.useWasmCache = false;
env.experimental_useCrossOriginStorage = false;
const onnxWasm = env.backends.onnx.wasm;
if (onnxWasm) onnxWasm.wasmPaths = { wasm: new URL(ortWasmUrl, self.location.href).href };

/** Whisper decodes in 30 s windows; language detection looks at the first one. */
const WINDOW_SECONDS = 30;

const post = (message: WorkerResponse): void => {
  self.postMessage(message);
};

let asr: AutomaticSpeechRecognitionPipeline | null = null;

async function gpuInfo(): Promise<GpuInfo> {
  if (!('gpu' in navigator)) return { available: false, reason: 'navigator.gpu is undefined' };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { available: false, reason: 'requestAdapter() returned null' };
  const { info, features, limits } = adapter;
  return {
    available: true,
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    isFallbackAdapter: info.isFallbackAdapter,
    shaderF16: features.has('shader-f16'),
    maxBufferSize: limits.maxBufferSize,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
  };
}

async function load(request: Extract<WorkerRequest, { type: 'load' }>): Promise<void> {
  await unload();
  const { id, revision } = MODELS[request.model];
  const started = performance.now();
  asr = await pipeline('automatic-speech-recognition', id, {
    revision,
    device: request.device,
    dtype: { encoder_model: request.dtype.encoder, decoder_model_merged: request.dtype.decoder },
    progress_callback: (progress: ProgressInfo) => {
      if (progress.status === 'progress') {
        post({
          type: 'progress',
          id: request.id,
          file: progress.file,
          loaded: progress.loaded,
          total: progress.total,
        });
      }
    },
  });
  post({ type: 'loaded', id: request.id, ms: performance.now() - started });
}

async function unload(): Promise<void> {
  const current = asr;
  asr = null;
  await current?.dispose();
}

interface WhisperGenerationConfig {
  decoder_start_token_id: number;
  lang_to_id: Record<string, number>;
}

/**
 * Transformers.js 4.3.1 doesn't detect the language: with none set it decodes as English. So run
 * one decoder step from <|startoftranscript|> on the first 30 s and take the most likely language
 * token, as openai/whisper's detect_language does.
 */
async function detectLanguage(pipe: AutomaticSpeechRecognitionPipeline, audio: Float32Array) {
  const { input_features } = (await pipe.processor(
    audio.subarray(0, WINDOW_SECONDS * ASR_SAMPLE_RATE),
  )) as { input_features: Tensor };
  const config = pipe.model.generation_config as unknown as WhisperGenerationConfig;
  const decoder_input_ids = new Tensor(
    'int64',
    BigInt64Array.from([BigInt(config.decoder_start_token_id)]),
    [1, 1],
  );
  const output = (await pipe.model({ input_features, decoder_input_ids })) as { logits: Tensor };
  const logits = output.logits.type === 'float32' ? output.logits : output.logits.to('float32');
  const scores = logits.data as Float32Array;
  let best = 'en';
  let bestScore = -Infinity;
  for (const [token, tokenId] of Object.entries(config.lang_to_id)) {
    const score = scores[tokenId] ?? -Infinity;
    if (score > bestScore) {
      bestScore = score;
      best = token.slice(2, -2); // "<|hi|>" -> "hi"
    }
  }
  return best;
}

async function transcribe(
  audio: Float32Array,
  task: AsrTask,
  forcedLanguage: string | null,
): Promise<Omit<Extract<WorkerResponse, { type: 'result' }>, 'type' | 'id'>> {
  const pipe = asr;
  if (!pipe) throw new Error('No model loaded.');
  const started = performance.now();
  let language = forcedLanguage;
  let detectMs: number | null = null;
  if (language === null) {
    language = await detectLanguage(pipe, audio);
    detectMs = performance.now() - started;
  }
  // One clip in, one output out (the array form is only for batched input).
  const { text } = (await pipe(audio, {
    language,
    task,
    chunk_length_s: WINDOW_SECONDS,
    stride_length_s: 5,
    return_timestamps: false,
  })) as { text: string };
  return { text: text.trim(), language, detectMs, ms: performance.now() - started };
}

async function handle(request: WorkerRequest): Promise<void> {
  switch (request.type) {
    case 'gpu-info':
      post({ type: 'gpu-info', id: request.id, info: await gpuInfo() });
      return;
    case 'load':
      await load(request);
      return;
    case 'transcribe':
      post({
        type: 'result',
        id: request.id,
        ...(await transcribe(request.audio, request.task, request.language)),
      });
      return;
    case 'unload':
      await unload();
      post({ type: 'unloaded', id: request.id });
      return;
  }
}

// One request at a time: loads and inference share the GPU and the ORT sessions.
let queue: Promise<void> = Promise.resolve();

self.addEventListener('message', (event: MessageEvent<unknown>) => {
  const request = event.data;
  if (!isWorkerRequest(request)) {
    console.warn('[tabscribe] ASR worker ignored an invalid request');
    return;
  }
  queue = queue.then(() =>
    handle(request).catch((error: unknown) => {
      const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      post({ type: 'error', id: request.id, message: message.slice(0, 1000) });
    }),
  );
});
