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
// - Nothing is downloaded unless the load request allows it (an explicit user action). Every
//   request Transformers.js makes goes through env.fetch below, which also pins it and answers
//   it from the cache when it can, so a cached start makes no network request.
//
// Threads: when the extension is cross-origin isolated, ORT's WASM backend starts its pthreads as
// module workers from this same script (import.meta.url) named "em-pthread". Those copies only run
// ORT's thread loop, so they skip this worker's message handler.
import {
  env,
  LogitsProcessorList,
  pipeline,
  Tensor,
  type AutomaticSpeechRecognitionPipeline,
  type ProgressInfo,
} from '@huggingface/transformers';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import { GeneratedNoRepeatNGram, Interrupt, logProbability, TokenLogprobs } from './decoding';
import { gpuInfo } from './gpu';
import { skipPaddingFrames, type WhisperFeatureExtractorLike } from './features';
import {
  compressionRatio,
  judge,
  maxNewTokens,
  NO_REPEAT_NGRAM_SIZE,
  NO_SPEECH_THRESHOLD,
  TEMPERATURES,
  type Attempt,
  type Verdict,
} from './guards';
import { HF_HOST, MODEL_CACHE, MODELS, pinnedUrl } from './models';
import {
  isWorkerRequest,
  type AsrTask,
  type AttemptSummary,
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
/** Set per load from its request. */
let downloadsAllowed = false;
const networkFetch = env.fetch;
// Every request Transformers.js makes, pinned to our revision (some of its probes name `main`),
// answered from the model cache when possible, and refused unless this load may download.
env.fetch = async (input, init) => {
  const url = pinnedUrl(String(input));
  const cached = await caches
    .open(MODEL_CACHE)
    .then((cache) => cache.match(url))
    .catch(() => undefined);
  if (cached) return cached;
  if (!downloadsAllowed) throw new Error(`ModelNotDownloaded: ${url} isn't cached`);
  return (await networkFetch(url, init)) as Response;
};
const onnxWasm = env.backends.onnx.wasm;
if (onnxWasm) onnxWasm.wasmPaths = { wasm: new URL(ortWasmUrl, self.location.href).href };

/** Whisper decodes in 30 s windows; language detection looks at the first one. */
const WINDOW_SECONDS = 30;

const post = (message: WorkerResponse): void => {
  self.postMessage(message);
};

let asr: AutomaticSpeechRecognitionPipeline | null = null;
/** ORT fixes its thread count when its WASM module starts, at this worker's first load. */
let ortThreads: number | null = null;

function startOrtWith(requested: number): number {
  // Without SharedArrayBuffer ORT runs on one thread whatever it's asked for.
  const threads = self.crossOriginIsolated ? requested : 1;
  if (ortThreads === null) {
    if (onnxWasm) onnxWasm.numThreads = threads;
    ortThreads = threads;
  } else if (ortThreads !== threads) {
    throw new Error(
      `ONNX Runtime already runs on ${String(ortThreads)} thread(s) in this worker; start a new worker for ${String(threads)}`,
    );
  }
  return threads;
}

async function load(request: Extract<WorkerRequest, { type: 'load' }>): Promise<void> {
  await unload();
  const threads = startOrtWith(request.threads);
  downloadsAllowed = request.allowDownload;
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
  allowPrecomputedEncoderOutput(asr);
  skipPaddingFrames(asr.processor.feature_extractor as unknown as WhisperFeatureExtractorLike);
  checkSpecialTokens(asr);
  const loaded = performance.now();
  await warmUp(asr);
  post({
    type: 'loaded',
    id: request.id,
    ms: loaded - started,
    threads,
    warmMs: performance.now() - loaded,
  });
}

/**
 * Whisper's `forward_params` in Transformers.js 4.3.1 omit `encoder_outputs`, so generate()
 * silently drops a precomputed encoder output and runs the encoder again. Allowing it on this
 * instance lets one encoder pass serve several decoder runs (profiling now; no-speech detection,
 * then decoding, in P3b).
 */
function allowPrecomputedEncoderOutput(pipe: AutomaticSpeechRecognitionPipeline): void {
  const model = pipe.model as unknown as { forward_params: string[] };
  if (!model.forward_params.includes('encoder_outputs')) {
    model.forward_params = [...model.forward_params, 'encoder_outputs'];
  }
}

async function unload(): Promise<void> {
  const current = asr;
  asr = null;
  await current?.dispose();
}

interface WhisperGenerationConfig {
  decoder_start_token_id: number;
  eos_token_id: number;
  lang_to_id: Record<string, number>;
  task_to_id: Record<AsrTask, number>;
  no_timestamps_token_id: number;
  prev_sot_token_id: number;
}

const whisperConfig = (pipe: AutomaticSpeechRecognitionPipeline): WhisperGenerationConfig =>
  pipe.model.generation_config as unknown as WhisperGenerationConfig;

/** openai/whisper's no-speech token directly follows <|startofprev|>, in every Whisper vocabulary. */
const noSpeechToken = (config: WhisperGenerationConfig): number => config.prev_sot_token_id + 1;

function checkSpecialTokens(pipe: AutomaticSpeechRecognitionPipeline): void {
  const config = whisperConfig(pipe);
  const name = pipe.tokenizer.decode([noSpeechToken(config)], { skip_special_tokens: false });
  if (name !== '<|nocaptions|>' && name !== '<|nospeech|>') {
    throw new Error(`Unexpected no-speech token ${JSON.stringify(name)}`);
  }
}

interface FirstStep {
  /** Most likely language ("en", "hi", ...). */
  language: string;
  /** Among languages only, as openai/whisper's detect_language. */
  englishProb: number;
  /** Over the whole vocabulary, as openai/whisper's no_speech_prob. */
  noSpeechProb: number;
}

/**
 * One decoder step from <|startoftranscript|>: the language and no-speech probabilities, as
 * openai/whisper reads them. Without `encoder_outputs`, the encoder runs first.
 */
async function firstStep(
  pipe: AutomaticSpeechRecognitionPipeline,
  input_features: Tensor,
  encoder_outputs?: Tensor,
): Promise<FirstStep> {
  const config = whisperConfig(pipe);
  const decoder_input_ids = new Tensor(
    'int64',
    BigInt64Array.from([BigInt(config.decoder_start_token_id)]),
    [1, 1],
  );
  const output = (await pipe.model({ input_features, encoder_outputs, decoder_input_ids })) as {
    logits: Tensor;
  };
  const logits = output.logits.type === 'float32' ? output.logits : output.logits.to('float32');
  const scores = logits.data as Float32Array;
  const languages = Object.entries(config.lang_to_id).map(([token, id]) => ({
    language: token.slice(2, -2), // "<|hi|>" -> "hi"
    score: scores[id] ?? -Infinity,
  }));
  const max = Math.max(...languages.map(({ score }) => score));
  const total = languages.reduce((sum, { score }) => sum + Math.exp(score - max), 0);
  const best = languages.reduce((a, b) => (b.score > a.score ? b : a));
  const english = languages.find(({ language }) => language === 'en');
  return {
    language: best.language,
    englishProb: english ? Math.exp(english.score - max) / total : 0,
    noSpeechProb: Math.exp(logProbability(scores, noSpeechToken(config))),
  };
}

/**
 * Transformers.js 4.3.1 doesn't detect the language: with none set it decodes as English. So take
 * the most likely language token from the first 30 s, as openai/whisper's detect_language does.
 */
async function detectLanguage(pipe: AutomaticSpeechRecognitionPipeline, audio: Float32Array) {
  const { input_features } = (await pipe.processor(
    audio.subarray(0, WINDOW_SECONDS * ASR_SAMPLE_RATE),
  )) as { input_features: Tensor };
  return (await firstStep(pipe, input_features)).language;
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

/** A raw ONNX Runtime tensor: the single-argument form of Transformers.js's Tensor constructor. */
type OrtTensor = Extract<ConstructorParameters<typeof Tensor>, [unknown]>[0];

/** The ONNX Runtime session Transformers.js keeps for the Whisper encoder. */
interface EncoderSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor | undefined>>;
}

/** <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> */
const PROMPT_TOKENS = 4;

/** Runs the encoder session once and returns its output for generate() to reuse. */
async function encode(
  pipe: AutomaticSpeechRecognitionPipeline,
  input_features: Tensor,
): Promise<Tensor> {
  const encoder = (
    pipe.model as unknown as { sessions: Record<string, EncoderSession | undefined> }
  ).sessions.model;
  if (!encoder) throw new Error('Whisper encoder session not found.');
  const encoded = await encoder.run({ input_features: input_features.ort_tensor });
  const lastHiddenState = encoded.last_hidden_state;
  if (!lastHiddenState) throw new Error('Encoder returned no last_hidden_state.');
  return new Tensor(lastHiddenState);
}

type Generate = (options: Record<string, unknown>) => Promise<unknown>;

/**
 * Below this, a segment is taken as not English (captions are English only). Measured in piece 4:
 * accented parliamentary English went as low as 0.27, Hindi as high as 0.14 (French and German
 * 0.08). 0.2 is near the geometric middle; the margin is thin on both sides.
 */
const ENGLISH_MIN_PROB = 0.2;

/** The four task tokens that start an English transcription without timestamps. */
const taskTokens = (config: WhisperGenerationConfig): number[] => [
  config.decoder_start_token_id,
  config.lang_to_id['<|en|>'] ?? 0,
  config.task_to_id.transcribe,
  config.no_timestamps_token_id,
];

interface Generated {
  attempt: Attempt;
  /** Text tokens, without special tokens. */
  tokens: number[];
}

/** One decode at `temperature` (0 is greedy) from `init`: the prompt and the task tokens. */
async function generateOnce(
  pipe: AutomaticSpeechRecognitionPipeline,
  inputs: { input_features: Tensor; encoder_outputs: Tensor },
  init: readonly number[],
  temperature: number,
  maxNew: number,
  interrupt: Interrupt,
): Promise<Generated> {
  const config = whisperConfig(pipe);
  const logprobs = new TokenLogprobs(temperature);
  const processors = new LogitsProcessorList();
  processors.push(new GeneratedNoRepeatNGram(NO_REPEAT_NGRAM_SIZE, init.length));
  processors.push(logprobs);
  const generate = pipe.model.generate.bind(pipe.model) as Generate;
  const output = (await generate({
    inputs: inputs.input_features,
    encoder_outputs: inputs.encoder_outputs,
    decoder_input_ids: [...init],
    max_new_tokens: maxNew,
    return_timestamps: false,
    // Sampling keeps Transformers.js's default top_k of 50; openai/whisper samples from all tokens.
    do_sample: temperature > 0,
    temperature: temperature > 0 ? temperature : 1,
    logits_processor: processors,
    stopping_criteria: [interrupt],
  })) as Tensor;
  const generated = ((output.tolist() as bigint[][])[0] ?? []).slice(init.length);
  const last = generated.at(-1);
  const endOfText = BigInt(config.eos_token_id);
  // Whisper's special tokens all come after <|endoftext|>.
  const tokens = generated.filter((token) => token < endOfText).map(Number);
  const text = pipe.tokenizer.decode(tokens, { skip_special_tokens: true }).trim();
  return {
    attempt: {
      text,
      temperature,
      avgLogprob: logprobs.average(last),
      compressionRatio: await compressionRatio(text),
      hitTokenCap: last !== endOfText && generated.length >= maxNew,
    },
    tokens,
  };
}

type Decoded = Omit<Extract<WorkerResponse, { type: 'decoded' }>, 'type' | 'id'>;

/**
 * One caption segment, after openai/whisper's transcribe(): one encoder pass, the language and
 * no-speech probabilities from the first step, then greedy decoding with the guards, retried hotter
 * on a loop or low confidence. Only text that passed (or was merely doubtful) is returned.
 */
async function decode(
  request: Extract<WorkerRequest, { type: 'decode' }>,
  interrupt: Interrupt,
): Promise<Decoded> {
  const pipe = asr;
  if (!pipe) throw new Error('No model loaded.');
  const config = whisperConfig(pipe);
  const started = performance.now();
  const { input_features } = (await pipe.processor(request.audio)) as { input_features: Tensor };
  const featured = performance.now();
  const encoder_outputs = await encode(pipe, input_features);
  const encoded = performance.now();
  const first = await firstStep(pipe, input_features, encoder_outputs);
  const attempts: AttemptSummary[] = [];
  const result = (outcome: Decoded['outcome'], promptTokens = 0, kept?: Generated): Decoded => ({
    outcome,
    text: kept?.attempt.text ?? '',
    tokens: kept?.tokens ?? [],
    language: first.language,
    englishProb: first.englishProb,
    noSpeechProb: first.noSpeechProb,
    promptTokens,
    attempts,
    featuresMs: featured - started,
    encoderMs: encoded - featured,
    ms: performance.now() - started,
  });

  if (first.englishProb < ENGLISH_MIN_PROB) {
    return result(first.noSpeechProb > NO_SPEECH_THRESHOLD ? 'no-speech' : 'not-english');
  }
  const prompt = request.prompt?.language === 'en' ? request.prompt.tokens : [];
  const init = [
    ...(prompt.length > 0 ? [config.prev_sot_token_id, ...prompt] : []),
    ...taskTokens(config),
  ];
  const maxNew = maxNewTokens(request.audio.length / ASR_SAMPLE_RATE);
  const temperatures = request.kind === 'partial' ? [0] : TEMPERATURES;
  const inputs = { input_features, encoder_outputs };
  let kept: Generated | undefined;
  let verdict: Verdict = 'ok';
  for (const temperature of temperatures) {
    const generated = await generateOnce(pipe, inputs, init, temperature, maxNew, interrupt);
    if (interrupt.requested) return result('interrupted', prompt.length);
    verdict = judge(generated.attempt, first.noSpeechProb);
    const { avgLogprob, compressionRatio: ratio, hitTokenCap } = generated.attempt;
    attempts.push({ temperature, avgLogprob, compressionRatio: ratio, hitTokenCap, verdict });
    kept = generated;
    if (verdict === 'ok' || verdict === 'no-speech') break;
  }
  const shown = verdict === 'ok' || verdict === 'low-confidence';
  return result(verdict, prompt.length, shown ? kept : undefined);
}

/** Half a second of silence through every stage, so the first real caption runs warm. */
async function warmUp(pipe: AutomaticSpeechRecognitionPipeline): Promise<void> {
  const { input_features } = (await pipe.processor(new Float32Array(ASR_SAMPLE_RATE / 2))) as {
    input_features: Tensor;
  };
  const encoder_outputs = await encode(pipe, input_features);
  await firstStep(pipe, input_features, encoder_outputs);
  const inputs = { input_features, encoder_outputs };
  await generateOnce(pipe, inputs, taskTokens(whisperConfig(pipe)), 0, 4, new Interrupt());
}

/**
 * Times the parts of one English decode separately. The encoder runs on its own session, then
 * generate() decodes from its output (it skips the encoder when `encoder_outputs` is given). The
 * encoder's output is read back to the CPU (only the decoder's KV cache stays on the GPU), so the
 * encoder time includes the work actually finishing.
 */
async function profile(
  audio: Float32Array,
): Promise<Omit<Extract<WorkerResponse, { type: 'profile' }>, 'type' | 'id'>> {
  const pipe = asr;
  if (!pipe) throw new Error('No model loaded.');
  const started = performance.now();
  const { input_features } = (await pipe.processor(audio)) as { input_features: Tensor };
  const featured = performance.now();
  const encoder_outputs = await encode(pipe, input_features);
  const encoderDone = performance.now();
  const generate = pipe.model.generate.bind(pipe.model) as Generate;
  const output = (await generate({
    inputs: input_features,
    encoder_outputs,
    language: 'en',
    task: 'transcribe',
    max_new_tokens: 224,
  })) as Tensor;
  const decoded = performance.now();
  const tokens = output.tolist() as bigint[][];
  const text = pipe.tokenizer.decode(tokens[0] ?? [], { skip_special_tokens: true }).trim();
  return {
    featuresMs: featured - started,
    encoderMs: encoderDone - featured,
    decoderMs: decoded - encoderDone,
    decoderSteps: Math.max(0, (output.dims[1] ?? 0) - PROMPT_TOKENS),
    text,
  };
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
    case 'profile':
      post({ type: 'profile', id: request.id, ...(await profile(request.audio)) });
      return;
    case 'decode': {
      const interrupt = interrupts.get(request.id) ?? new Interrupt();
      try {
        post({ type: 'decoded', id: request.id, ...(await decode(request, interrupt)) });
      } finally {
        interrupts.delete(request.id);
      }
      return;
    }
    case 'interrupt':
      return;
    case 'unload':
      await unload();
      post({ type: 'unloaded', id: request.id });
      return;
  }
}

// One request at a time: loads and inference share the GPU and the ORT sessions.
let queue: Promise<void> = Promise.resolve();
/**
 * Decodes queued or running, by request id. generate() awaits ORT between steps, so an interrupt
 * message is handled while a decode runs.
 */
const interrupts = new Map<number, Interrupt>();

function onRequest(event: MessageEvent<unknown>): void {
  const request = event.data;
  if (!isWorkerRequest(request)) {
    console.warn('[tabscribe] ASR worker ignored an invalid request');
    return;
  }
  if (request.type === 'interrupt') {
    const interrupt = interrupts.get(request.id);
    if (interrupt) interrupt.requested = true;
    return;
  }
  if (request.type === 'decode') interrupts.set(request.id, new Interrupt());
  queue = queue.then(() =>
    handle(request).catch((error: unknown) => {
      const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      post({ type: 'error', id: request.id, message: message.slice(0, 1000) });
    }),
  );
}

// ORT's pthreads (see the top of this file) leave the message channel to ORT.
if (!self.name.startsWith('em-pthread')) self.addEventListener('message', onRequest);
