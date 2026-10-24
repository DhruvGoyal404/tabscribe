// ONNX Runtime's WASM backend runs on several threads only when the page is cross-origin isolated
// (it needs SharedArrayBuffer); otherwise it runs on one, whatever it's asked for.

/** Most threads a load request may ask for. */
export const MAX_THREADS = 16;

export const isThreadCount = (value: unknown): value is number =>
  Number.isSafeInteger(value) && Number(value) >= 1 && Number(value) <= MAX_THREADS;

/**
 * Half the cores, at most 4: enough for the encoder's matrix multiplications to scale, while
 * leaving the rest for the tab being captioned (often a video) and the rest of the browser.
 */
export function defaultThreads(crossOriginIsolated: boolean, hardwareConcurrency: number): number {
  if (!crossOriginIsolated) return 1;
  return Math.max(1, Math.min(4, Math.floor(hardwareConcurrency / 2)));
}
