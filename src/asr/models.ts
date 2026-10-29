// Whisper models the ASR worker may load. Each is pinned to a commit, so the worker only ever
// fetches files from that exact revision. Sizes come from the Hugging Face API at that revision
// (`/api/models/<id>?blobs=true`) and let the UI show the download size before anything is fetched.
// The onnx-community repos don't declare a licence; the weights are converted from
// openai/whisper-*, which is Apache-2.0.

export const DEVICES = ['webgpu', 'wasm'] as const;
export type Device = (typeof DEVICES)[number];

/** Transformers.js dtypes these repos ship files for (q8 is the `_quantized` file). */
export const DTYPES = ['fp32', 'fp16', 'q8', 'int8', 'uint8', 'q4', 'bnb4'] as const;
export type Dtype = (typeof DTYPES)[number];

/** Encoder and decoder dtypes are chosen separately: they behave differently per device. */
export interface DtypeSpec {
  encoder: Dtype;
  decoder: Dtype;
}

export interface ModelInfo {
  id: string;
  revision: string;
  /** onnx/encoder_model{suffix}.onnx */
  encoderBytes: Record<Dtype, number>;
  /** onnx/decoder_model_merged{suffix}.onnx */
  decoderBytes: Record<Dtype, number>;
  /** config, generation_config, preprocessor_config, tokenizer and tokenizer_config JSON. */
  configBytes: number;
}

export const MODELS = {
  'whisper-tiny': {
    id: 'onnx-community/whisper-tiny',
    revision: 'ff4177021cc41f7db950912b73ea4fdf7d01d8e7',
    encoderBytes: { fp32: 32904992, fp16: 16519192, q8: 10124990, int8: 10124977, uint8: 10124990, q4: 9020667, bnb4: 8578451 },
    decoderBytes: { fp32: 118553827, fp16: 59593896, q8: 30719241, int8: 30719241, uint8: 30719271, q4: 86713702, bnb4: 86124414 },
    configBytes: 2769503,
  },
  'whisper-base': {
    id: 'onnx-community/whisper-base',
    revision: '1846881b6b3a3024392c1eea3ad983695bc23925',
    encoderBytes: { fp32: 82468078, fp16: 41332612, q8: 23201314, int8: 23201297, uint8: 23201314, q4: 18772451, bnb4: 17593091 },
    decoderBytes: { fp32: 208521528, fp16: 104727818, q8: 53693315, int8: 53693315, uint8: 53693343, q4: 123602419, bnb4: 122030467 },
    configBytes: 2769562,
  },
  'whisper-small': {
    id: 'onnx-community/whisper-small',
    revision: '36050c46d777d46dc4b5f43f6d90574fc38f8732',
    encoderBytes: { fp32: 352825870, fp16: 176607756, q8: 92326160, int8: 92326127, uint8: 92326160, q4: 66182104, bnb4: 60874216 },
    decoderBytes: { fp32: 615324301, fp16: 308583076, q8: 156750845, int8: 156750845, uint8: 156750906, q4: 233149327, bnb4: 226073167 },
    configBytes: 2769608,
  },
} as const satisfies Record<string, ModelInfo>; // prettier-ignore

export type ModelKey = keyof typeof MODELS;
export const MODEL_KEYS = Object.keys(MODELS) as ModelKey[];

const DTYPE_SUFFIX: Record<Dtype, string> = {
  fp32: '',
  fp16: '_fp16',
  q8: '_quantized',
  int8: '_int8',
  uint8: '_uint8',
  q4: '_q4',
  bnb4: '_bnb4',
};

export const HF_HOST = 'https://huggingface.co/';
/** Cache API bucket Transformers.js stores model files in, keyed by modelFileUrls. */
export const MODEL_CACHE = 'transformers-cache';

/** Whether every file a load would fetch is already cached (false if the Cache API fails). */
export async function isCached(key: ModelKey, dtype: DtypeSpec): Promise<boolean> {
  try {
    const cache = await caches.open(MODEL_CACHE);
    const hits = await Promise.all(modelFileUrls(key, dtype).map((url) => cache.match(url)));
    return hits.every((hit) => hit !== undefined);
  } catch {
    return false;
  }
}

/** Every file a load fetches, as the URLs Transformers.js requests (and uses as cache keys). */
export function modelFileUrls(key: ModelKey, dtype: DtypeSpec): string[] {
  const { id, revision } = MODELS[key];
  const base = `${HF_HOST}${id}/resolve/${revision}/`;
  return [
    'config.json',
    'generation_config.json',
    'preprocessor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    `onnx/encoder_model${DTYPE_SUFFIX[dtype.encoder]}.onnx`,
    `onnx/decoder_model_merged${DTYPE_SUFFIX[dtype.decoder]}.onnx`,
  ].map((file) => base + file);
}

/**
 * `url` at the pinned revision. Transformers.js 4.3.1 probes some files (whether they exist, and
 * their size) without passing the revision, so those requests name `main`, a moving branch.
 */
export function pinnedUrl(url: string): string {
  for (const { id, revision } of Object.values(MODELS) as ModelInfo[]) {
    const unpinned = `${HF_HOST}${id}/resolve/main/`;
    if (url.startsWith(unpinned))
      return `${HF_HOST}${id}/resolve/${revision}/${url.slice(unpinned.length)}`;
  }
  return url;
}

export function downloadBytes(key: ModelKey, dtype: DtypeSpec): number {
  const model: ModelInfo = MODELS[key];
  return model.configBytes + model.encoderBytes[dtype.encoder] + model.decoderBytes[dtype.decoder];
}

export const isModelKey = (value: unknown): value is ModelKey =>
  typeof value === 'string' && Object.hasOwn(MODELS, value);
export const isDevice = (value: unknown): value is Device =>
  (DEVICES as readonly unknown[]).includes(value);
export const isDtype = (value: unknown): value is Dtype =>
  (DTYPES as readonly unknown[]).includes(value);
