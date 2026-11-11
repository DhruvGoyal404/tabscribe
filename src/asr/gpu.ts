// What WebGPU offers here, for choosing the caption backend. Runs in any extension page or worker.
import type { GpuInfo } from './protocol';

/**
 * Chrome ignores powerPreference on Windows and logs a warning each time it's passed there
 * (crbug.com/369219127), so it's only asked for elsewhere, where it picks a dual-GPU Mac's faster GPU.
 */
export const IGNORES_POWER_PREFERENCE = navigator.userAgent.includes('Windows');
const ADAPTER_OPTIONS: GPURequestAdapterOptions = IGNORES_POWER_PREFERENCE
  ? {}
  : { powerPreference: 'high-performance' };

export async function gpuInfo(): Promise<GpuInfo> {
  if (!('gpu' in navigator)) return { available: false, reason: 'navigator.gpu is undefined' };
  const adapter = await navigator.gpu.requestAdapter(ADAPTER_OPTIONS);
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
