import { crx } from '@crxjs/vite-plugin';
import { defineConfig } from 'vite';
import manifest from './manifest.config.ts';

export default defineConfig({
  plugins: [crx({ manifest })],
  // The AudioWorklet is imported with `?worker&url` and loaded via audioWorklet.addModule,
  // which expects a module script.
  worker: { format: 'es' },
  build: {
    target: 'chrome116',
    rollupOptions: {
      // The offscreen document isn't referenced by the manifest, so it needs its own entry.
      input: { offscreen: 'src/offscreen/index.html' },
    },
  },
});
