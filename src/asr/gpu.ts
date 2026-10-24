// What WebGPU offers here, for choosing the caption backend. Runs in any extension page or worker.
import type { GpuInfo } from './protocol';

export async function gpuInfo(): Promise<GpuInfo> {
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
