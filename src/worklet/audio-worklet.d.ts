// Minimal AudioWorkletGlobalScope declarations; TypeScript's DOM lib doesn't model this scope.
// https://webaudio.github.io/web-audio-api/#AudioWorkletGlobalScope

declare const sampleRate: number;

/** The parts of MessagePort the processors use: their node's port, and ports sent over it. */
interface WorkletPort {
  postMessage(message: unknown, transfer?: unknown[]): void;
  onmessage: ((event: { data: unknown }) => void) | null;
}

declare abstract class AudioWorkletProcessor {
  readonly port: WorkletPort;
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
