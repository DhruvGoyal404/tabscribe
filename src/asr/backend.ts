// Which backend captions run on, and why. P3a/P3b chose whisper-base on WebGPU fp16, with WASM q8
// on up to 4 threads as the fallback (docs/spikes/p3b-streaming.md). The popup plans before
// starting, so it can show the download size and ask first; the offscreen document loads exactly
// that plan.
import { defaultThreads, isThreadCount } from './threads';
import { downloadBytes, isDevice, type Device, type DtypeSpec } from './models';
import type { GpuInfo } from './protocol';

export const CAPTION_MODEL = 'whisper-base';

/**
 * The restricted previous-text prompt (context.ts) is off for captions: it was mixed on the bench
 * once the overlap de-dup worked. The tuning piece measures it again on continuous speech.
 */
export const CAPTION_PROMPTING = false;

export interface CaptionPlan {
  device: Device;
  dtype: DtypeSpec;
  /** ONNX Runtime WASM threads (WebGPU runs few ops on the CPU, but the count still applies). */
  threads: number;
  /** Why this backend, for the popup. */
  reason: string;
}

const MAX_REASON_LENGTH = 300;

/** The only dtypes captions load, per device. */
const DTYPES: Record<Device, DtypeSpec> = {
  webgpu: { encoder: 'fp16', decoder: 'fp16' },
  wasm: { encoder: 'q8', decoder: 'q8' },
};

export interface PlanInputs {
  gpu: GpuInfo;
  /** A previous WebGPU load failure on this browser, if any. */
  webgpuFailure: string | null;
  crossOriginIsolated: boolean;
  hardwareConcurrency: number;
}

export function planBackend({
  gpu,
  webgpuFailure,
  crossOriginIsolated,
  hardwareConcurrency,
}: PlanInputs): CaptionPlan {
  const threads = defaultThreads(crossOriginIsolated, hardwareConcurrency);
  const wasm = (reason: string): CaptionPlan => ({
    device: 'wasm',
    dtype: DTYPES.wasm,
    threads,
    reason: reason.slice(0, MAX_REASON_LENGTH),
  });
  if (webgpuFailure !== null) return wasm(`WebGPU failed to load before: ${webgpuFailure}`);
  if (!gpu.available) return wasm(`WebGPU unavailable: ${gpu.reason}`);
  if (gpu.isFallbackAdapter) return wasm('WebGPU only has a software adapter');
  if (!gpu.shaderF16) return wasm('the WebGPU adapter lacks shader-f16');
  const adapter = [gpu.vendor, gpu.architecture].filter(Boolean).join(' ') || 'adapter';
  return { device: 'webgpu', dtype: DTYPES.webgpu, threads, reason: `WebGPU (${adapter})` };
}

export const planBytes = (plan: CaptionPlan): number => downloadBytes(CAPTION_MODEL, plan.dtype);

export function describePlan(plan: CaptionPlan): string {
  const { encoder, decoder } = plan.dtype;
  const where = plan.device === 'webgpu' ? 'WebGPU' : `WASM, ${String(plan.threads)} threads`;
  return `${CAPTION_MODEL} ${encoder}/${decoder} on ${where}`;
}

/** A plan as planBackend makes them: the caption dtypes for its device, nothing else. */
export function isCaptionPlan(value: unknown): value is CaptionPlan {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as Record<string, unknown>;
  if (!isDevice(record.device)) return false;
  const expected = DTYPES[record.device];
  const dtype = record.dtype as Record<string, unknown> | null;
  return (
    Object.keys(record).sort().join(',') === 'device,dtype,reason,threads' &&
    typeof dtype === 'object' &&
    dtype !== null &&
    Object.keys(dtype).sort().join(',') === 'decoder,encoder' &&
    dtype.encoder === expected.encoder &&
    dtype.decoder === expected.decoder &&
    isThreadCount(record.threads) &&
    typeof record.reason === 'string' &&
    record.reason.length > 0 &&
    record.reason.length <= MAX_REASON_LENGTH
  );
}
