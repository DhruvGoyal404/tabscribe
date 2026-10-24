// Dev-only ASR benchmark. Spawns the production ASR worker, runs the fixtures through it and
// reports load time, real-time factor and accuracy per (model, device, dtype). Results are kept in
// localStorage so a page reload (for a cold worker with a warm cache) doesn't lose them.
import {
  DEVICES,
  downloadBytes,
  DTYPES,
  isDevice,
  isDtype,
  isModelKey,
  MODEL_KEYS,
  MODELS,
  modelFileUrls,
  type Device,
  type DtypeSpec,
  type ModelKey,
} from '../asr/models';
import {
  isWorkerResponse,
  type AsrTask,
  type DecodePrompt,
  type GpuInfo,
  type WorkerRequest,
  type WorkerResponse,
} from '../asr/protocol';
import { ASR_SAMPLE_RATE, downmixToMono, Resampler } from '../asr/resample';
import { defaultThreads, MAX_THREADS } from '../asr/threads';
import {
  PROFILE_OFFSETS,
  PROFILE_SECONDS,
  secondsToSamples,
  segmentOffsets,
  summarise,
  type CallTiming,
  type CostRow,
} from './cost';
import { FIXTURES, type Fixture } from './fixtures';
import { characterErrorRate, wordErrorRate } from './metrics';
import { DEFAULT_SEGMENTER } from '../vad/segmenter';
import { decodeStream, type StreamRun } from './stream';
import { runVad, type VadRun } from './vad-run';

/** Cache API bucket Transformers.js stores model files in. */
const MODEL_CACHE = 'transformers-cache';
/** Fixtures are decoded at a typical tab-capture rate and resampled like live audio will be. */
const CAPTURE_RATE = 48_000;
const RESULTS_KEY = 'tabscribe-bench-results';
const COSTS_KEY = 'tabscribe-bench-costs';
const STREAM_KEY = 'tabscribe-bench-stream';

interface BenchRow {
  model: ModelKey;
  device: Device;
  dtype: DtypeSpec;
  /** ORT threads; missing in rows stored before the thread selector existed (they ran on 1). */
  threads?: number;
  cachedBeforeLoad: boolean;
  loadMs: number;
  enRtfFirst: number;
  enRtfSecond: number;
  enWer: number;
  hiRtf: number;
  hiCer: number;
  hiWer: number;
  detectedEn: string;
  detectedHi: string;
  detectMs: number;
  outputs: { en: string; hi: string; hiTranslate: string };
}

function element<T extends HTMLElement>(id: string, type: new () => T): T {
  const found = document.getElementById(id);
  if (!(found instanceof type)) throw new Error(`#${id} missing`);
  return found;
}

const ui = {
  environment: element('environment', HTMLElement),
  model: element('model', HTMLSelectElement),
  device: element('device', HTMLSelectElement),
  encoder: element('encoder', HTMLSelectElement),
  decoder: element('decoder', HTMLSelectElement),
  size: element('size', HTMLElement),
  load: element('load', HTMLButtonElement),
  run: element('run', HTMLButtonElement),
  unload: element('unload', HTMLButtonElement),
  clearCache: element('clear-cache', HTMLButtonElement),
  progress: element('progress', HTMLElement),
  error: element('error', HTMLElement),
  copy: element('copy', HTMLButtonElement),
  clearResults: element('clear-results', HTMLButtonElement),
  results: element('results', HTMLElement),
  outputs: element('outputs', HTMLElement),
  profile: element('profile', HTMLButtonElement),
  clearCosts: element('clear-costs', HTMLButtonElement),
  costs: element('costs', HTMLElement),
  costTexts: element('cost-texts', HTMLElement),
  threads: element('threads', HTMLSelectElement),
  vadKind: element('vad-kind', HTMLSelectElement),
  vadMax: element('vad-max', HTMLInputElement),
  runVad: element('run-vad', HTMLButtonElement),
  vadResults: element('vad-results', HTMLElement),
  streamMax: element('stream-max', HTMLInputElement),
  runStream: element('run-stream', HTMLButtonElement),
  clearStream: element('clear-stream', HTMLButtonElement),
  streamResults: element('stream-results', HTMLElement),
  streamTexts: element('stream-texts', HTMLElement),
};

// --- Worker client ------------------------------------------------------------------------------

let nextId = 0;
const pending = new Map<
  number,
  { resolve: (response: WorkerResponse) => void; reject: (error: Error) => void }
>();
const fileProgress = new Map<string, { loaded: number; total: number }>();

function onWorkerMessage(event: MessageEvent<unknown>): void {
  const response = event.data;
  if (!isWorkerResponse(response)) return;
  if (response.type === 'progress') {
    fileProgress.set(response.file, { loaded: response.loaded, total: response.total });
    let loaded = 0;
    let total = 0;
    for (const file of fileProgress.values()) {
      loaded += file.loaded;
      total += file.total;
    }
    ui.progress.textContent = `Downloading ${mb(loaded)} / ${mb(total)} (${response.file})`;
    return;
  }
  const waiter = pending.get(response.id);
  pending.delete(response.id);
  if (response.type === 'error') waiter?.reject(new Error(response.message));
  else waiter?.resolve(response);
}

function spawnWorker(): Worker {
  const spawned = new Worker(new URL('../asr/asr.worker.ts', import.meta.url), { type: 'module' });
  spawned.addEventListener('message', onWorkerMessage);
  spawned.addEventListener('error', (event) => {
    showError(`Worker error: ${event.message}`);
  });
  return spawned;
}

let worker = spawnWorker();

/** A fresh worker, so ORT can start with another thread count. */
function restartWorker(): void {
  worker.terminate();
  for (const waiter of pending.values()) waiter.reject(new Error('Worker restarted'));
  pending.clear();
  worker = spawnWorker();
}

function send<T extends WorkerResponse['type']>(
  expected: T,
  make: (id: number) => WorkerRequest,
  transfer: Transferable[] = [],
): Promise<Extract<WorkerResponse, { type: T }>> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, {
      resolve: (response) => {
        if (response.type === expected) resolve(response as Extract<WorkerResponse, { type: T }>);
        else reject(new Error(`Expected ${expected}, got ${response.type}`));
      },
      reject,
    });
    worker.postMessage(make(id), transfer);
  });
}

async function transcribe(audio: Float32Array, task: AsrTask, language: string | null) {
  const copy = audio.slice();
  return send('result', (id) => ({ type: 'transcribe', id, audio: copy, task, language }), [
    copy.buffer,
  ]);
}

// --- Helpers ------------------------------------------------------------------------------------

const mb = (bytes: number): string => `${(bytes / 1e6).toFixed(1)} MB`;
const pct = (rate: number): string => `${(rate * 100).toFixed(1)}%`;
const rtf = (ms: number, samples: number): number => ms / 1000 / (samples / ASR_SAMPLE_RATE);

function showError(message: string): void {
  ui.error.textContent = message;
}

function setBusy(busy: boolean, loaded = currentModel !== null): void {
  ui.load.disabled = busy;
  ui.run.disabled = busy || !loaded;
  ui.unload.disabled = busy || !loaded;
  ui.profile.disabled = busy || !loaded;
  ui.clearCache.disabled = busy;
  ui.runVad.disabled = busy;
  ui.runStream.disabled = busy || !loaded;
  for (const select of [ui.model, ui.device, ui.encoder, ui.decoder, ui.threads]) {
    select.disabled = busy;
  }
}

function selected(): { model: ModelKey; device: Device; dtype: DtypeSpec } {
  const { value: model } = ui.model;
  const { value: device } = ui.device;
  const { value: encoder } = ui.encoder;
  const { value: decoder } = ui.decoder;
  if (!isModelKey(model) || !isDevice(device) || !isDtype(encoder) || !isDtype(decoder)) {
    throw new Error('Invalid selection');
  }
  return { model, device, dtype: { encoder, decoder } };
}

const autoThreads = defaultThreads(self.crossOriginIsolated, navigator.hardwareConcurrency);

function selectedThreads(): number {
  const { value } = ui.threads;
  return value === 'auto' ? autoThreads : Number(value);
}

/** Without cross-origin isolation ORT runs on one thread whatever it's asked for. */
const effectiveThreads = (requested: number): number => (self.crossOriginIsolated ? requested : 1);

async function cachedFlags(urls: string[]): Promise<boolean[]> {
  try {
    const cache = await caches.open(MODEL_CACHE);
    return await Promise.all(urls.map(async (url) => (await cache.match(url)) !== undefined));
  } catch {
    return urls.map(() => false);
  }
}

// --- Fixtures -----------------------------------------------------------------------------------

const captured = new Map<Fixture['key'], Float32Array>();
const decoded = new Map<Fixture['key'], Float32Array>();

/** The fixture as tab capture would deliver it: mono at 48 kHz. */
async function fixtureCapture(fixture: Fixture): Promise<Float32Array> {
  const cached = captured.get(fixture.key);
  if (cached) return cached;
  const bytes = await (await fetch(fixture.url)).arrayBuffer();
  const context = new OfflineAudioContext({
    numberOfChannels: 1,
    length: 1,
    sampleRate: CAPTURE_RATE,
  });
  const buffer = await context.decodeAudioData(bytes);
  const mono = downmixToMono(
    Array.from({ length: buffer.numberOfChannels }, (_, i) => buffer.getChannelData(i)),
  );
  captured.set(fixture.key, mono);
  return mono;
}

/** 16 kHz for Whisper, through our resampler as live audio will be. */
async function fixtureAudio(fixture: Fixture): Promise<Float32Array> {
  const cached = decoded.get(fixture.key);
  if (cached) return cached;
  const resampler = new Resampler({ inputRate: CAPTURE_RATE });
  const head = resampler.push(await fixtureCapture(fixture));
  const tail = resampler.flush();
  const audio = new Float32Array(head.length + tail.length);
  audio.set(head);
  audio.set(tail, head.length);
  decoded.set(fixture.key, audio);
  return audio;
}

function fixture(key: Fixture['key']): Fixture {
  const found = FIXTURES.find((candidate) => candidate.key === key);
  if (!found) throw new Error(`Missing fixture ${key}`);
  return found;
}

// --- Results ------------------------------------------------------------------------------------

function loadStored<T>(key: string): T[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(key) ?? '[]');
    return Array.isArray(parsed) ? (parsed as T[]) : [];
  } catch {
    return [];
  }
}

function store(key: string, values: readonly unknown[]): void {
  try {
    localStorage.setItem(key, JSON.stringify(values));
  } catch {
    // Results stay on screen; only persistence across reloads is lost.
  }
}

const rows = loadStored<BenchRow>(RESULTS_KEY);
const costs = loadStored<CostRow>(COSTS_KEY);

function saveRows(): void {
  store(RESULTS_KEY, rows);
}

const threadsLabel = (threads: number): string =>
  `${String(threads)} thread${threads === 1 ? '' : 's'}`;

const configLabel = (model: ModelKey, device: Device, dtype: DtypeSpec, threads = 1): string =>
  `${model} · ${device} · ${dtype.encoder}/${dtype.decoder} · ${threadsLabel(threads)}`;

const label = (row: BenchRow): string => configLabel(row.model, row.device, row.dtype, row.threads);

function cells(row: BenchRow): string[] {
  return [
    row.model,
    `${row.device} (${threadsLabel(row.threads ?? 1)})`,
    `${row.dtype.encoder} / ${row.dtype.decoder}`,
    row.cachedBeforeLoad ? 'cached' : 'download',
    (row.loadMs / 1000).toFixed(1),
    row.enRtfFirst.toFixed(3),
    row.enRtfSecond.toFixed(3),
    pct(row.enWer),
    row.hiRtf.toFixed(3),
    pct(row.hiCer),
    pct(row.hiWer),
    `${row.detectedEn} / ${row.detectedHi}`,
    row.detectMs.toFixed(0),
  ];
}

function renderRows(): void {
  ui.results.replaceChildren(
    ...rows.map((row) => {
      const tr = document.createElement('tr');
      for (const text of cells(row)) {
        const td = document.createElement('td');
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    }),
  );
  const translationReference = fixture('hi').translationReference ?? '';
  ui.outputs.replaceChildren(
    ...rows.map((row) => {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = `Outputs: ${label(row)}`;
      details.append(summary);
      for (const [title, text] of [
        ['English transcript', row.outputs.en],
        ['Hindi transcript', row.outputs.hi],
        ['Hindi → English translation', row.outputs.hiTranslate],
        ['Reference translation (FLoRes)', translationReference],
      ] as const) {
        const heading = document.createElement('h3');
        heading.textContent = title;
        const body = document.createElement('p');
        body.textContent = text;
        details.append(heading, body);
      }
      return details;
    }),
  );
}

// --- Environment --------------------------------------------------------------------------------

let gpu: GpuInfo | null = null;

function environment(): [string, string][] {
  const gpuText = !gpu
    ? 'checking…'
    : gpu.available
      ? [
          `${gpu.vendor} ${gpu.architecture} ${gpu.device} ${gpu.description}`.trim() ||
            '(no info)',
          gpu.isFallbackAdapter ? 'FALLBACK adapter' : 'hardware adapter',
          `shader-f16 ${gpu.shaderF16 ? 'yes' : 'no'}`,
          `maxBufferSize ${mb(gpu.maxBufferSize)}`,
          `maxStorageBufferBindingSize ${mb(gpu.maxStorageBufferBindingSize)}`,
        ].join(' · ')
      : `unavailable (${gpu.reason})`;
  return [
    ['Browser', navigator.userAgent],
    ['WebGPU in worker', gpuText],
    ['CPU threads', String(navigator.hardwareConcurrency)],
    ['crossOriginIsolated', `${String(self.crossOriginIsolated)} (false = single-threaded WASM)`],
    ['Default ORT threads', threadsLabel(autoThreads)],
  ];
}

function renderEnvironment(): void {
  ui.environment.replaceChildren(
    ...environment().flatMap(([term, value]) => {
      const dt = document.createElement('dt');
      dt.textContent = term;
      const dd = document.createElement('dd');
      dd.textContent = value;
      return [dt, dd];
    }),
  );
}

function costCells(row: CostRow): string[] {
  return [
    row.config,
    String(row.seconds),
    row.featuresMs.toFixed(0),
    row.encoderMs.toFixed(0),
    row.decoderMs.toFixed(0),
    row.decoderSteps.toFixed(0),
    row.msPerStep.toFixed(1),
    row.totalMs.toFixed(0),
    row.callRtf.toFixed(3),
  ];
}

function renderCosts(): void {
  ui.costs.replaceChildren(
    ...costs.map((row) => {
      const tr = document.createElement('tr');
      for (const text of costCells(row)) {
        const td = document.createElement('td');
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    }),
  );
  ui.costTexts.replaceChildren(
    ...costs.map((row) => {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = `Decoded text: ${row.config}, ${String(row.seconds)} s segments`;
      const list = document.createElement('ol');
      for (const text of row.texts) {
        const item = document.createElement('li');
        item.textContent = text;
        list.append(item);
      }
      details.append(summary, list);
      return details;
    }),
  );
}

const vadRuns: { fixture: Fixture['key']; run: VadRun }[] = [];

const secondsAt = (sample: number): string => (sample / ASR_SAMPLE_RATE).toFixed(2);

function vadCells({ fixture: key, run }: (typeof vadRuns)[number]): string[] {
  const finals = run.events.filter((event) => event.type === 'final');
  return [
    key,
    run.fallbackReason ? `${run.vad} (fallback: ${run.fallbackReason})` : run.vad,
    String(finals.length),
    String(run.stats.partials),
    `${((100 * run.stats.speechFrames) / Math.max(1, run.stats.frames)).toFixed(0)}%`,
    run.stats.msPerFrame.toFixed(2),
    run.ms.toFixed(0),
    finals
      .map((e) => `${secondsAt(e.startSample)}–${secondsAt(e.endSample)} (${e.reason})`)
      .join(', '),
  ];
}

function renderVad(): void {
  ui.vadResults.replaceChildren(
    ...vadRuns.map((entry) => {
      const tr = document.createElement('tr');
      for (const text of vadCells(entry)) {
        const td = document.createElement('td');
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    }),
  );
}

async function runVadOnFixtures(): Promise<void> {
  const { value: kind } = ui.vadKind;
  if (kind !== 'silero' && kind !== 'energy') return;
  const maxSegmentMs = Number(ui.vadMax.value);
  showError('');
  setBusy(true);
  try {
    for (const entry of FIXTURES) {
      ui.progress.textContent = `VAD: ${entry.key}…`;
      const run = await runVad(await fixtureCapture(entry), CAPTURE_RATE, kind, {
        maxSegmentMs,
      });
      vadRuns.push({ fixture: entry.key, run });
      renderVad();
    }
    ui.progress.textContent = 'VAD done.';
  } catch (error) {
    showError(String(error));
    ui.progress.textContent = '';
  } finally {
    setBusy(false);
  }
}

// --- Streaming captions -------------------------------------------------------------------------

interface StreamRow {
  config: string;
  fixture: Fixture['key'];
  maxSegmentMs: number;
  prompting: boolean;
  /** Of the joined captions; null where captions don't apply (Hindi: English only). */
  wer: number | null;
  stats: StreamRun['stats'];
  transcript: string;
}

const streamRows = loadStored<StreamRow>(STREAM_KEY);

function decodeSegment(audio: Float32Array, prompt: DecodePrompt | null) {
  const copy = audio.slice();
  return send('decoded', (id) => ({ type: 'decode', id, audio: copy, kind: 'final', prompt }), [
    copy.buffer,
  ]);
}

function streamCells(row: StreamRow): string[] {
  const { stats } = row;
  return [
    row.config,
    row.fixture,
    `${String(row.maxSegmentMs / 1000)} s`,
    row.prompting ? 'on' : 'off',
    String(stats.segments),
    String(stats.forcedCuts),
    String(stats.prompted),
    row.wer === null ? '—' : pct(row.wer),
    String(stats.loopsCaught),
    String(stats.loopsShown),
    String(stats.retries),
    Object.entries(stats.outcomes)
      .map(([outcome, count]) => `${outcome} ${String(count)}`)
      .join(', '),
    (stats.decodeMs / 1000).toFixed(1),
  ];
}

const streamLabel = (row: StreamRow): string =>
  `${row.config}, ${row.fixture}, ${String(row.maxSegmentMs / 1000)} s max, prompt ${row.prompting ? 'on' : 'off'}`;

function renderStream(): void {
  ui.streamResults.replaceChildren(
    ...streamRows.map((row) => {
      const tr = document.createElement('tr');
      for (const text of streamCells(row)) {
        const td = document.createElement('td');
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    }),
  );
  ui.streamTexts.replaceChildren(
    ...streamRows.map((row) => {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = `Captions: ${streamLabel(row)}`;
      const text = document.createElement('p');
      text.textContent = row.transcript || '(none shown)';
      details.append(summary, text);
      return details;
    }),
  );
}

/**
 * Each fixture through the VAD, then its finals through the guarded decode in order, with and
 * without the previous-text prompt, at the default maximum segment and at a shorter one that forces
 * cuts (the prompt only applies after a forced cut).
 */
async function runStream(): Promise<void> {
  const loaded = currentModel;
  if (!loaded) return;
  const maxima = [...new Set([DEFAULT_SEGMENTER.maxSegmentMs, Number(ui.streamMax.value)])];
  showError('');
  setBusy(true);
  try {
    const config = configLabel(loaded.model, loaded.device, loaded.dtype, loaded.threads);
    for (const entry of FIXTURES) {
      for (const maxSegmentMs of maxima) {
        const name = `${entry.key}, ${String(maxSegmentMs)} ms max`;
        ui.progress.textContent = `Streaming ${name}: VAD…`;
        const vad = await runVad(await fixtureCapture(entry), CAPTURE_RATE, 'silero', {
          maxSegmentMs,
        });
        const finals = vad.events.filter((event) => event.type === 'final');
        for (const prompting of [false, true]) {
          ui.progress.textContent = `Streaming ${name}, prompt ${prompting ? 'on' : 'off'}: decoding ${String(finals.length)} segments…`;
          const run = await decodeStream(finals, decodeSegment, prompting);
          streamRows.push({
            config,
            fixture: entry.key,
            maxSegmentMs,
            prompting,
            wer:
              entry.language === 'en'
                ? wordErrorRate(entry.reference, run.transcript, 'en').rate
                : null,
            stats: run.stats,
            transcript: run.transcript,
          });
          store(STREAM_KEY, streamRows);
          renderStream();
        }
      }
    }
    ui.progress.textContent = `Streaming done: ${config}.`;
  } catch (error) {
    showError(String(error));
    ui.progress.textContent = '';
  } finally {
    setBusy(false);
  }
}

function markdown(): string {
  const header = [
    'Model', 'Device', 'Enc / Dec', 'Load', 'Load (s)', 'EN RTF 1st', 'EN RTF 2nd', 'EN WER',
    'HI RTF', 'HI CER', 'HI WER', 'Detect EN / HI', 'Detect (ms)',
  ]; // prettier-ignore
  const line = (values: string[]) => `| ${values.join(' | ')} |`;
  return [
    '### Environment',
    ...environment().map(([term, value]) => `- ${term}: ${value}`),
    '',
    '### Results',
    line(header),
    line(header.map(() => '---')),
    ...rows.map((row) => line(cells(row))),
    '',
    '### Per-call cost',
    line([
      'Config',
      'Segment (s)',
      'Features (ms)',
      'Encoder (ms)',
      'Decoder (ms)',
      'Decoder steps',
      'ms / step',
      'Total (ms)',
      'Call RTF',
    ]),
    line(Array.from({ length: 9 }, () => '---')),
    ...costs.map((row) => line(costCells(row))),
    '',
    '### VAD',
    line([
      'Fixture',
      'VAD',
      'Segments',
      'Partials',
      'Speech',
      'ms / frame',
      'Stream (ms)',
      'Segments',
    ]),
    line(Array.from({ length: 8 }, () => '---')),
    ...vadRuns.map((entry) => line(vadCells(entry))),
    '',
    '### Streaming captions',
    line([
      'Config',
      'Fixture',
      'Max segment',
      'Prompt',
      'Segments',
      'Forced cuts',
      'Prompted',
      'WER',
      'Loops caught',
      'Loops shown',
      'Retries',
      'Outcomes',
      'Decode (s)',
    ]),
    line(Array.from({ length: 13 }, () => '---')),
    ...streamRows.map((row) => line(streamCells(row))),
    '',
    ...streamRows.map((row) => `- ${streamLabel(row)}: ${row.transcript}`),
    '',
    '### Outputs',
    ...rows.flatMap((row) => [
      '',
      `#### ${label(row)}`,
      `- EN: ${row.outputs.en}`,
      `- HI: ${row.outputs.hi}`,
      `- HI→EN: ${row.outputs.hiTranslate}`,
    ]),
  ].join('\n');
}

// --- Actions ------------------------------------------------------------------------------------

let currentModel: { model: ModelKey; device: Device; dtype: DtypeSpec; threads: number; loadMs: number; cachedBeforeLoad: boolean } | null = null; // prettier-ignore
/** The thread count ORT started with in the current worker, once it has loaded a model. */
let workerThreads: number | null = null;

async function updateSize(): Promise<void> {
  const { model, dtype } = selected();
  const urls = modelFileUrls(model, dtype);
  const flags = await cachedFlags(urls);
  const cachedCount = flags.filter(Boolean).length;
  const total = downloadBytes(model, dtype);
  const revision = MODELS[model].revision.slice(0, 7);
  if (cachedCount === urls.length) {
    ui.size.textContent = `All ${String(urls.length)} files cached (${mb(total)}), revision ${revision}. Loading downloads nothing.`;
    ui.load.textContent = 'Load model (from cache)';
  } else {
    ui.size.textContent = `Download: up to ${mb(total)} from huggingface.co (${String(cachedCount)} of ${String(urls.length)} files already cached), revision ${revision}.`;
    ui.load.textContent = `Load model (downloads up to ${mb(total)})`;
  }
}

/** Loads into the current worker, restarting it first if ORT runs there on another count. */
async function loadInto(
  choice: ReturnType<typeof selected>,
  requested: number,
): Promise<NonNullable<typeof currentModel>> {
  const cachedBeforeLoad = (await cachedFlags(modelFileUrls(choice.model, choice.dtype))).every(
    Boolean,
  );
  ui.progress.textContent = cachedBeforeLoad ? 'Loading from cache…' : 'Starting download…';
  currentModel = null;
  if (workerThreads !== null && workerThreads !== effectiveThreads(requested)) restartWorker();
  // ORT fixes the count as soon as the worker starts loading, even if the load then fails.
  workerThreads = effectiveThreads(requested);
  const { ms, threads } = await send('loaded', (id) => ({
    type: 'load',
    id,
    ...choice,
    threads: requested,
  }));
  workerThreads = threads;
  const loaded = { ...choice, threads, loadMs: ms, cachedBeforeLoad };
  currentModel = loaded;
  ui.progress.textContent = `Loaded ${choice.model} on ${choice.device} (${threadsLabel(threads)}) in ${(ms / 1000).toFixed(1)} s.`;
  return loaded;
}

async function loadModel(): Promise<void> {
  showError('');
  setBusy(true, false);
  fileProgress.clear();
  try {
    await loadInto(selected(), selectedThreads());
  } catch (error) {
    showError(String(error));
    ui.progress.textContent = '';
  } finally {
    setBusy(false);
    void updateSize();
  }
}

/** The loaded model on the selected thread count: reloaded from cache if the selection changed. */
async function withSelectedThreads(
  loaded: NonNullable<typeof currentModel>,
): Promise<NonNullable<typeof currentModel>> {
  const requested = selectedThreads();
  if (effectiveThreads(requested) === loaded.threads) return loaded;
  const { model, device, dtype } = loaded;
  return loadInto({ model, device, dtype }, requested);
}

async function runBenchmark(): Promise<void> {
  const current = currentModel;
  if (!current) return;
  showError('');
  setBusy(true);
  try {
    const loaded = await withSelectedThreads(current);
    ui.progress.textContent = 'Decoding fixtures…';
    const en = fixture('en');
    const hi = fixture('hi');
    const enAudio = await fixtureAudio(en);
    const hiAudio = await fixtureAudio(hi);
    const steps = 6;
    const step = async (n: number, name: string, run: () => ReturnType<typeof transcribe>) => {
      ui.progress.textContent = `Running ${String(n)}/${String(steps)}: ${name}…`;
      return run();
    };
    const enFirst = await step(1, 'English, forced en (first run)', () =>
      transcribe(enAudio, 'transcribe', 'en'),
    );
    const enSecond = await step(2, 'English, forced en (second run)', () =>
      transcribe(enAudio, 'transcribe', 'en'),
    );
    const enDetect = await step(3, 'English, auto-detect', () =>
      transcribe(enAudio, 'transcribe', null),
    );
    const hiForced = await step(4, 'Hindi, forced hi', () =>
      transcribe(hiAudio, 'transcribe', 'hi'),
    );
    const hiDetect = await step(5, 'Hindi, auto-detect', () =>
      transcribe(hiAudio, 'transcribe', null),
    );
    const hiTranslate = await step(6, 'Hindi → English translate', () =>
      transcribe(hiAudio, 'translate', 'hi'),
    );
    const row: BenchRow = {
      model: loaded.model,
      device: loaded.device,
      dtype: loaded.dtype,
      threads: loaded.threads,
      cachedBeforeLoad: loaded.cachedBeforeLoad,
      loadMs: loaded.loadMs,
      enRtfFirst: rtf(enFirst.ms, enAudio.length),
      enRtfSecond: rtf(enSecond.ms, enAudio.length),
      enWer: wordErrorRate(en.reference, enSecond.text, 'en').rate,
      hiRtf: rtf(hiForced.ms, hiAudio.length),
      hiCer: characterErrorRate(hi.reference, hiForced.text, 'hi').rate,
      hiWer: wordErrorRate(hi.reference, hiForced.text, 'hi').rate,
      detectedEn: enDetect.language,
      detectedHi: hiDetect.language,
      detectMs: ((enDetect.detectMs ?? 0) + (hiDetect.detectMs ?? 0)) / 2,
      outputs: { en: enSecond.text, hi: hiForced.text, hiTranslate: hiTranslate.text },
    };
    rows.push(row);
    saveRows();
    renderRows();
    ui.progress.textContent = `Done: ${label(row)}.`;
  } catch (error) {
    showError(String(error));
    ui.progress.textContent = '';
  } finally {
    setBusy(false);
  }
}

async function profileCalls(): Promise<void> {
  let loaded = currentModel;
  if (!loaded) return;
  showError('');
  setBusy(true);
  try {
    loaded = await withSelectedThreads(loaded);
    const audio = await fixtureAudio(fixture('en'));
    const profileOne = (segment: Float32Array) => {
      const copy = segment.slice();
      return send('profile', (id) => ({ type: 'profile', id, audio: copy }), [copy.buffer]);
    };
    ui.progress.textContent = 'Per-call cost: warm-up…';
    await profileOne(audio.subarray(0, secondsToSamples(PROFILE_SECONDS[0])));
    const config = configLabel(loaded.model, loaded.device, loaded.dtype, loaded.threads);
    for (const seconds of PROFILE_SECONDS) {
      const length = secondsToSamples(seconds);
      const timings: CallTiming[] = [];
      for (const offset of segmentOffsets(audio.length, length, PROFILE_OFFSETS)) {
        ui.progress.textContent = `Per-call cost: ${String(seconds)} s segment at ${(offset / ASR_SAMPLE_RATE).toFixed(1)} s…`;
        timings.push(await profileOne(audio.subarray(offset, offset + length)));
      }
      costs.push(summarise(config, seconds, timings));
      store(COSTS_KEY, costs);
      renderCosts();
    }
    ui.progress.textContent = `Per-call cost done: ${config}.`;
  } catch (error) {
    showError(String(error));
    ui.progress.textContent = '';
  } finally {
    setBusy(false);
  }
}

async function unloadModel(): Promise<void> {
  setBusy(true);
  try {
    await send('unloaded', (id) => ({ type: 'unload', id }));
    currentModel = null;
    ui.progress.textContent = 'Unloaded.';
  } catch (error) {
    showError(String(error));
  } finally {
    setBusy(false);
  }
}

async function clearModelCache(): Promise<void> {
  if (!confirm('Delete every cached model file? The next load downloads again.')) return;
  await caches.delete(MODEL_CACHE);
  ui.progress.textContent = 'Model cache cleared.';
  await updateSize();
}

// --- Wiring -------------------------------------------------------------------------------------

function fillSelect(select: HTMLSelectElement, values: readonly string[], initial: string): void {
  select.replaceChildren(
    ...values.map((value) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      return option;
    }),
  );
  select.value = initial;
}

fillSelect(ui.model, MODEL_KEYS, 'whisper-base');
fillSelect(ui.device, DEVICES, 'webgpu');
fillSelect(ui.encoder, DTYPES, 'fp32');
fillSelect(ui.decoder, DTYPES, 'q4');
fillSelect(ui.vadKind, ['silero', 'energy'], 'silero');
fillSelect(
  ui.threads,
  [
    'auto',
    ...[1, 2, 4, 8, 12, 16]
      .filter((n) => n <= Math.min(MAX_THREADS, navigator.hardwareConcurrency))
      .map(String),
  ],
  'auto',
);
const autoOption = ui.threads.querySelector('option[value="auto"]');
if (autoOption) autoOption.textContent = `auto (${String(autoThreads)})`;

for (const select of [ui.model, ui.device, ui.encoder, ui.decoder]) {
  select.addEventListener('change', () => void updateSize());
}
ui.load.addEventListener('click', () => void loadModel());
ui.run.addEventListener('click', () => void runBenchmark());
ui.unload.addEventListener('click', () => void unloadModel());
ui.clearCache.addEventListener('click', () => void clearModelCache());
ui.copy.addEventListener('click', () => {
  void navigator.clipboard.writeText(markdown()).then(
    () => (ui.progress.textContent = 'Results copied as Markdown.'),
    (error: unknown) => {
      showError(`Copy failed: ${String(error)}`);
    },
  );
});
ui.profile.addEventListener('click', () => void profileCalls());
ui.runVad.addEventListener('click', () => void runVadOnFixtures());
ui.runStream.addEventListener('click', () => void runStream());
ui.clearStream.addEventListener('click', () => {
  streamRows.length = 0;
  store(STREAM_KEY, streamRows);
  renderStream();
});
ui.clearCosts.addEventListener('click', () => {
  costs.length = 0;
  store(COSTS_KEY, costs);
  renderCosts();
});
ui.clearResults.addEventListener('click', () => {
  rows.length = 0;
  saveRows();
  renderRows();
});

renderEnvironment();
renderRows();
renderCosts();
renderStream();
setBusy(false);
void updateSize();
void send('gpu-info', (id) => ({ type: 'gpu-info', id })).then(
  (response) => {
    gpu = response.info;
    renderEnvironment();
  },
  (error: unknown) => {
    gpu = { available: false, reason: String(error) };
    renderEnvironment();
  },
);
