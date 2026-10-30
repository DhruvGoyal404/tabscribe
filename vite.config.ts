import { crx, type CrxPlugin } from '@crxjs/vite-plugin';
import { defineConfig } from 'vite';
import manifest from './manifest.config.ts';

// crxjs makes every dynamic script (`?iife`) web-accessible to all sites. The overlay doesn't need
// that: chrome.scripting.executeScript injects extension files directly. A web-accessible file
// would let any page detect the extension, so this drops it. `post`, to run after crxjs's own.
const OVERLAY = 'src/overlay/';
const noWebAccessibleOverlay: CrxPlugin = {
  name: 'tabscribe:no-web-accessible-overlay',
  enforce: 'post',
  renderCrxManifest(rendered) {
    const resources = (rendered.web_accessible_resources ?? [])
      .map((entry) => ({
        ...entry,
        resources: entry.resources.filter((r) => !r.startsWith(OVERLAY)),
      }))
      .filter((entry) => entry.resources.length > 0);
    if (resources.length > 0) rendered.web_accessible_resources = resources;
    else delete rendered.web_accessible_resources;
    return rendered;
  },
};

// `vite build --mode bench` (npm run build:bench) adds the dev-only ASR bench page and writes to
// dist-bench/, so the production dist/ never contains the bench, its fixtures or ONNX Runtime.
export default defineConfig(({ mode }) => {
  const bench = mode === 'bench';
  return {
    plugins: [crx({ manifest }), noWebAccessibleOverlay],
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
