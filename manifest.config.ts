import { defineManifest } from '@crxjs/vite-plugin';
import pkg from './package.json' with { type: 'json' };

const icons = {
  16: 'icons/icon-16.png',
  32: 'icons/icon-32.png',
  48: 'icons/icon-48.png',
  128: 'icons/icon-128.png',
};

export default defineManifest({
  manifest_version: 3,
  name: 'Tabscribe',
  version: pkg.version,
  description: pkg.description,
  // chrome.runtime.getContexts (finding the offscreen document) needs Chrome 116+.
  minimum_chrome_version: '116',
  icons,
  action: {
    default_title: 'Tabscribe',
    default_popup: 'src/popup/index.html',
    default_icon: icons,
  },
  options_ui: {
    page: 'src/options/index.html',
    open_in_tab: true,
  },
  background: {
    service_worker: 'src/background/index.ts',
    type: 'module',
  },
  permissions: ['activeTab', 'tabCapture', 'offscreen', 'scripting', 'storage'],
  content_security_policy: {
    // connect-src: our own files plus Hugging Face, where the pinned Whisper weights live.
    // huggingface.co serves the JSON configs; the ONNX weights redirect to its CDN, whose host
    // depends on region and storage backend (e.g. us.aws.cdn.hf.co), hence *.hf.co.
    extension_pages: [
      "script-src 'self' 'wasm-unsafe-eval'",
      "object-src 'self'",
      "connect-src 'self' https://huggingface.co https://*.hf.co",
    ]
      .join('; ')
      .concat(';'),
  },
});
