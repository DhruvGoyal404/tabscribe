import { describe, expect, it } from 'vitest';
import { describePlan, isCaptionPlan, planBackend, planBytes, type PlanInputs } from './backend';
import type { GpuInfo } from './protocol';

const gpu: GpuInfo = {
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
const inputs = (overrides: Partial<PlanInputs> = {}): PlanInputs => ({
  gpu,
  webgpuFailure: null,
  crossOriginIsolated: true,
  hardwareConcurrency: 12,
  ...overrides,
});

describe('planBackend', () => {
  it('uses WebGPU fp16 when the adapter has shader-f16', () => {
    const plan = planBackend(inputs());
    expect(plan).toEqual({
      device: 'webgpu',
      dtype: { encoder: 'fp16', decoder: 'fp16' },
      threads: 4,
      reason: 'WebGPU (nvidia ampere)',
    });
    expect(describePlan(plan)).toBe('whisper-base fp16/fp16 on WebGPU');
    expect(planBytes(plan)).toBe(148_829_992);
  });

  it.each([
    ['no adapter', { gpu: { available: false, reason: 'requestAdapter() returned null' } as const }, 'WebGPU unavailable: requestAdapter() returned null'],
    ['a software adapter', { gpu: { ...gpu, isFallbackAdapter: true } }, 'WebGPU only has a software adapter'],
    ['no shader-f16', { gpu: { ...gpu, shaderF16: false } }, 'the WebGPU adapter lacks shader-f16'],
    ['an earlier WebGPU failure', { webgpuFailure: 'Error: out of memory' }, 'WebGPU failed to load before: Error: out of memory'],
  ])('falls back to WASM q8 with %s', (_, overrides: Partial<PlanInputs>, reason) => {
    const plan = planBackend(inputs(overrides));
    expect(plan).toEqual({ device: 'wasm', dtype: { encoder: 'q8', decoder: 'q8' }, threads: 4, reason });
    expect(describePlan(plan)).toBe('whisper-base q8/q8 on WASM, 4 threads');
    expect(planBytes(plan)).toBe(79_664_191);
  }); // prettier-ignore

  it('runs WASM on one thread without cross-origin isolation', () => {
    expect(planBackend(inputs({ gpu: { ...gpu, shaderF16: false }, crossOriginIsolated: false })).threads).toBe(1);
  }); // prettier-ignore

  it('keeps a long failure reason within bounds', () => {
    const plan = planBackend(inputs({ webgpuFailure: 'x'.repeat(1000) }));
    expect(isCaptionPlan(plan)).toBe(true);
  });
});

describe('isCaptionPlan', () => {
  it('accepts the plans planBackend makes', () => {
    expect(isCaptionPlan(planBackend(inputs()))).toBe(true);
    expect(isCaptionPlan(planBackend(inputs({ webgpuFailure: 'x' })))).toBe(true);
  });

  it.each([
    ['null', null],
    ['another dtype for the device', { device: 'webgpu', dtype: { encoder: 'fp32', decoder: 'fp32' }, threads: 4, reason: 'r' }],
    ['the WASM dtype on WebGPU', { device: 'webgpu', dtype: { encoder: 'q8', decoder: 'q8' }, threads: 4, reason: 'r' }],
    ['an unknown device', { device: 'webnn', dtype: { encoder: 'fp16', decoder: 'fp16' }, threads: 4, reason: 'r' }],
    ['too many threads', { device: 'wasm', dtype: { encoder: 'q8', decoder: 'q8' }, threads: 64, reason: 'r' }],
    ['an empty reason', { device: 'wasm', dtype: { encoder: 'q8', decoder: 'q8' }, threads: 4, reason: '' }],
    ['a model key', { device: 'wasm', dtype: { encoder: 'q8', decoder: 'q8' }, threads: 4, reason: 'r', model: 'whisper-small' }],
  ])('rejects %s', (_, value) => {
    expect(isCaptionPlan(value)).toBe(false);
  }); // prettier-ignore
});
