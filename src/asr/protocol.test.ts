import { describe, expect, it } from 'vitest';
import { downloadBytes, MODEL_KEYS, MODELS, modelFileUrls, pinnedUrl } from './models';
import {
  isLanguageCode,
  isWorkerRequest,
  isWorkerResponse,
  MAX_AUDIO_SAMPLES,
  MAX_WINDOW_SAMPLES,
} from './protocol';

const audio = new Float32Array(16_000);
const dtype = { encoder: 'fp32', decoder: 'q4' };
const prompt = { tokens: [440, 1029, 13], language: 'en' };
const attempt = {
  temperature: 0,
  avgLogprob: -0.3,
  compressionRatio: 1.2,
  hitTokenCap: false,
  verdict: 'ok',
};
const decoded = {
  type: 'decoded',
  id: 5,
  outcome: 'ok',
  text: 'Hello.',
  tokens: [2425, 13],
  language: 'en',
  englishProb: 0.98,
  noSpeechProb: 0.01,
  promptTokens: 0,
  attempts: [attempt],
  featuresMs: 20,
  encoderMs: 300,
  ms: 900,
};

describe('isWorkerRequest', () => {
  it.each([
    { type: 'gpu-info', id: 0 },
    { type: 'unload', id: 3 },
    { type: 'load', id: 1, model: 'whisper-base', device: 'webgpu', dtype, threads: 1, allowDownload: true },
    { type: 'load', id: 1, model: 'whisper-tiny', device: 'wasm', dtype: { encoder: 'q8', decoder: 'q8' }, threads: 4, allowDownload: true },
    { type: 'transcribe', id: 2, audio, task: 'transcribe', language: null },
    { type: 'transcribe', id: 2, audio, task: 'transcribe', language: 'hi' },
    { type: 'transcribe', id: 2, audio, task: 'translate', language: 'hi' },
    { type: 'profile', id: 4, audio },
    { type: 'profile', id: 4, audio: new Float32Array(MAX_WINDOW_SAMPLES) },
    { type: 'decode', id: 5, audio, kind: 'final', prompt: null },
    { type: 'decode', id: 5, audio, kind: 'partial', prompt },
    { type: 'interrupt', id: 5 },
  ])('accepts %o', (request) => {
    expect(isWorkerRequest(request)).toBe(true);
  });

  it.each([
    ['null', null],
    ['a missing id', { type: 'gpu-info' }],
    ['a negative id', { type: 'gpu-info', id: -1 }],
    ['an unknown type', { type: 'download', id: 1 }],
    ['extra keys', { type: 'unload', id: 1, force: true }],
    ['an unpinned model', { type: 'load', id: 1, model: 'whisper-large-v3', device: 'webgpu', dtype, threads: 1, allowDownload: true }],
    ['a repo id instead of a key', { type: 'load', id: 1, model: 'onnx-community/whisper-base', device: 'webgpu', dtype, threads: 1, allowDownload: true }],
    ['a revision override', { type: 'load', id: 1, model: 'whisper-base', device: 'webgpu', dtype, threads: 1, allowDownload: true, revision: 'main' }],
    ['an unknown device', { type: 'load', id: 1, model: 'whisper-base', device: 'webnn', dtype, threads: 1, allowDownload: true }],
    ['a string dtype', { type: 'load', id: 1, model: 'whisper-base', device: 'wasm', dtype: 'q8', threads: 1, allowDownload: true }],
    ['an unknown dtype', { type: 'load', id: 1, model: 'whisper-base', device: 'wasm', dtype: { encoder: 'q2', decoder: 'q8' }, threads: 1, allowDownload: true }],
    ['no thread count', { type: 'load', id: 1, model: 'whisper-base', device: 'wasm', dtype, allowDownload: true }],
    ['no download flag', { type: 'load', id: 1, model: 'whisper-base', device: 'wasm', dtype, threads: 1 }],
    ['a string download flag', { type: 'load', id: 1, model: 'whisper-base', device: 'wasm', dtype, threads: 1, allowDownload: 'yes' }],
    ['zero threads', { type: 'load', id: 1, model: 'whisper-base', device: 'wasm', dtype, threads: 0, allowDownload: true }],
    ['too many threads', { type: 'load', id: 1, model: 'whisper-base', device: 'wasm', dtype, threads: 64, allowDownload: true }],
    ['audio as a plain array', { type: 'transcribe', id: 2, audio: [0, 0], task: 'transcribe', language: null }],
    ['empty audio', { type: 'transcribe', id: 2, audio: new Float32Array(0), task: 'transcribe', language: null }],
    ['too much audio', { type: 'transcribe', id: 2, audio: new Float32Array(MAX_AUDIO_SAMPLES + 1), task: 'transcribe', language: null }],
    ['an unknown task', { type: 'transcribe', id: 2, audio, task: 'summarise', language: null }],
    ['a language name', { type: 'transcribe', id: 2, audio, task: 'transcribe', language: 'hindi' }],
    ['a missing language', { type: 'transcribe', id: 2, audio, task: 'transcribe' }],
    ['a profile longer than one window', { type: 'profile', id: 4, audio: new Float32Array(MAX_WINDOW_SAMPLES + 1) }],
    ['a profile with options', { type: 'profile', id: 4, audio, language: 'en' }],
    ['a decode longer than one window', { type: 'decode', id: 5, audio: new Float32Array(MAX_WINDOW_SAMPLES + 1), kind: 'final', prompt: null }],
    ['an unknown decode kind', { type: 'decode', id: 5, audio, kind: 'draft', prompt: null }],
    ['a decode without a prompt key', { type: 'decode', id: 5, audio, kind: 'final' }],
    ['an empty prompt', { type: 'decode', id: 5, audio, kind: 'final', prompt: { tokens: [], language: 'en' } }],
    ['a prompt with a fractional token', { type: 'decode', id: 5, audio, kind: 'final', prompt: { tokens: [1.5], language: 'en' } }],
    ['a prompt token outside the vocabulary', { type: 'decode', id: 5, audio, kind: 'final', prompt: { tokens: [51_865], language: 'en' } }],
    ['a prompt over 223 tokens', { type: 'decode', id: 5, audio, kind: 'final', prompt: { tokens: new Array<number>(224).fill(1), language: 'en' } }],
    ['a prompt without a language', { type: 'decode', id: 5, audio, kind: 'final', prompt: { tokens: [1] } }],
    ['an interrupt with extra keys', { type: 'interrupt', id: 5, now: true }],
  ])('rejects %s', (_label, request) => {
    expect(isWorkerRequest(request)).toBe(false);
  });
}); // prettier-ignore

describe('isWorkerResponse', () => {
  const gpu = {
    available: true,
    vendor: 'nvidia',
    architecture: 'ampere',
    device: '',
    description: '',
    isFallbackAdapter: false,
    shaderF16: true,
    maxBufferSize: 2 ** 31,
    maxStorageBufferBindingSize: 2 ** 30,
  };

  it.each([
    { type: 'gpu-info', id: 0, info: gpu },
    { type: 'gpu-info', id: 0, info: { available: false, reason: 'navigator.gpu is undefined' } },
    { type: 'progress', id: 1, file: 'onnx/encoder_model.onnx', loaded: 10, total: 100 },
    { type: 'loaded', id: 1, ms: 1234.5, threads: 4, warmMs: 300 },
    decoded,
    { ...decoded, outcome: 'interrupted', text: '', tokens: [], attempts: [] },
    { ...decoded, attempts: [attempt, { ...attempt, temperature: 0.2, avgLogprob: -Infinity, verdict: 'loop' }] },
    { type: 'result', id: 2, text: 'hello', language: 'en', detectMs: null, ms: 900 },
    { type: 'result', id: 2, text: 'नमस्ते', language: 'hi', detectMs: 120, ms: 900 },
    { type: 'profile', id: 4, featuresMs: 12, encoderMs: 800, decoderMs: 450, decoderSteps: 31, text: 'hi' },
    { type: 'unloaded', id: 3 },
    { type: 'error', id: 3, message: 'boom' },
  ])('accepts %o', (response) => {
    expect(isWorkerResponse(response)).toBe(true);
  });

  it.each([
    ['gpu info with a missing field', { type: 'gpu-info', id: 0, info: { ...gpu, shaderF16: undefined } }],
    ['negative progress', { type: 'progress', id: 1, file: 'x', loaded: -1, total: 100 }],
    ['NaN timing', { type: 'loaded', id: 1, ms: NaN, threads: 1, warmMs: 1 }],
    ['a fractional thread count', { type: 'loaded', id: 1, ms: 1, threads: 1.5, warmMs: 1 }],
    ['a load without its warm-up time', { type: 'loaded', id: 1, ms: 1, threads: 1 }],
    ['an unknown outcome', { ...decoded, outcome: 'maybe' }],
    ['a probability over 1', { ...decoded, noSpeechProb: 1.5 }],
    ['a positive log-probability', { ...decoded, attempts: [{ ...attempt, avgLogprob: 0.1 }] }],
    ['a NaN log-probability', { ...decoded, attempts: [{ ...attempt, avgLogprob: NaN }] }],
    ['an attempt with an outcome as verdict', { ...decoded, attempts: [{ ...attempt, verdict: 'interrupted' }] }],
    ['too many attempts', { ...decoded, attempts: new Array(9).fill(attempt) }],
    ['a fractional prompt count', { ...decoded, promptTokens: 1.5 }],
    ['a result without a language', { type: 'result', id: 2, text: 'hi', detectMs: null, ms: 1 }],
    ['a non-string error', { type: 'error', id: 3, message: 42 }],
    ['fractional decoder steps', { type: 'profile', id: 4, featuresMs: 1, encoderMs: 1, decoderMs: 1, decoderSteps: 1.5, text: '' }],
  ])('rejects %s', (_label, response) => {
    expect(isWorkerResponse(response)).toBe(false);
  });
}); // prettier-ignore

describe('isLanguageCode', () => {
  it('accepts Whisper codes only', () => {
    expect(['en', 'hi', 'haw'].every(isLanguageCode)).toBe(true);
    expect(['', 'EN', 'english', 'e', '<|en|>', null].some(isLanguageCode)).toBe(false);
  });
});

describe('model catalogue', () => {
  it('pins a request for main to the catalogue revision', () => {
    const base = 'https://huggingface.co/onnx-community/whisper-base/resolve/';
    expect(pinnedUrl(`${base}main/tokenizer_config.json`)).toBe(
      `${base}${MODELS['whisper-base'].revision}/tokenizer_config.json`,
    );
    expect(pinnedUrl(`${base}main/onnx/encoder_model_fp16.onnx`)).toBe(
      `${base}${MODELS['whisper-base'].revision}/onnx/encoder_model_fp16.onnx`,
    );
  });

  it('leaves pinned and unrelated URLs alone', () => {
    const [pinned = ''] = modelFileUrls('whisper-tiny', { encoder: 'q8', decoder: 'q8' });
    expect(pinnedUrl(pinned)).toBe(pinned);
    const other = 'https://huggingface.co/someone/whisper-base/resolve/main/config.json';
    expect(pinnedUrl(other)).toBe(other);
  });

  it('pins every model to a full commit SHA', () => {
    for (const key of MODEL_KEYS) {
      expect(MODELS[key].revision).toMatch(/^[0-9a-f]{40}$/);
      expect(MODELS[key].id).toMatch(/^onnx-community\/whisper-(tiny|base|small)$/);
    }
  });

  it('builds file URLs on the pinned revision only', () => {
    const urls = modelFileUrls('whisper-base', { encoder: 'fp16', decoder: 'q4' });
    expect(urls).toHaveLength(7);
    for (const url of urls) {
      expect(url.startsWith(`https://huggingface.co/onnx-community/whisper-base/resolve/${MODELS['whisper-base'].revision}/`)).toBe(true);
    }
    expect(urls).toContain(`https://huggingface.co/onnx-community/whisper-base/resolve/${MODELS['whisper-base'].revision}/onnx/encoder_model_fp16.onnx`);
    expect(urls.some((url) => url.endsWith('/onnx/decoder_model_merged_q4.onnx'))).toBe(true);
    expect(modelFileUrls('whisper-tiny', { encoder: 'q8', decoder: 'q8' }).filter((url) => url.includes('_quantized.onnx'))).toHaveLength(2);
  }); // prettier-ignore

  it('sums the download size from the pinned file sizes', () => {
    const model = MODELS['whisper-tiny'];
    expect(downloadBytes('whisper-tiny', { encoder: 'fp32', decoder: 'q4' })).toBe(
      model.configBytes + model.encoderBytes.fp32 + model.decoderBytes.q4,
    );
    // Smallest useful config is tens of MB; the largest is about 1 GB.
    expect(downloadBytes('whisper-tiny', { encoder: 'q8', decoder: 'q8' })).toBeLessThan(50e6);
    expect(downloadBytes('whisper-small', { encoder: 'fp32', decoder: 'fp32' })).toBeGreaterThan(
      9e8,
    );
  });
});
