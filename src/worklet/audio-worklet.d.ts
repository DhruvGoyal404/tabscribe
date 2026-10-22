// Minimal AudioWorkletGlobalScope declarations; TypeScript's DOM lib doesn't model this scope.
// https://webaudio.github.io/web-audio-api/#AudioWorkletGlobalScope

declare const sampleRate: number;

declare abstract class AudioWorkletProcessor {
  readonly port: { postMessage(message: unknown): void };
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}

declare function registerProcessor(
  name: string,
  processorCtor: new (options?: unknown) => AudioWorkletProcessor,
): void;
