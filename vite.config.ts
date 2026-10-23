import { crx } from '@crxjs/vite-plugin';
import { defineConfig } from 'vite';
import manifest from './manifest.config.ts';

// `vite build --mode bench` (npm run build:bench) adds the dev-only ASR bench page and writes to
// dist-bench/, so the production dist/ never contains the bench, its fixtures or ONNX Runtime.
export default defineConfig(({ mode }) => {
  const bench = mode === 'bench';
  return {
    plugins: [crx({ manifest })],
    // The AudioWorklet is imported with `?worker&url` and loaded via audioWorklet.addModule,
    // which expects a module script. Module workers also let the ASR worker use import.meta.url.
    worker: { format: 'es' },
    build: {
      target: 'chrome116',
      outDir: bench ? 'dist-bench' : 'dist',
      rollupOptions: {
        // The offscreen document isn't referenced by the manifest, so it needs its own entry.
        input: {
          offscreen: 'src/offscreen/index.html',
          ...(bench ? { bench: 'src/bench/index.html' } : {}),
        },
      },
    },
  };
});
