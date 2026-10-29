#!/usr/bin/env node
// Checks the production build in dist/ after `npm run build`:
// - the manifest keeps exactly the approved permissions, no host permissions, the approved CSP and
//   cross-origin isolation;
// - nothing from the dev-only bench (page, fixtures) leaked in, and no Whisper weights;
// - the caption workers, ONNX Runtime and the Silero VAD model are there.
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIST = fileURLToPath(new URL('../dist/', import.meta.url));
const PERMISSIONS = ['activeTab', 'tabCapture', 'offscreen', 'scripting', 'storage'];
const CSP =
  "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'; " +
  "connect-src 'self' https://huggingface.co https://*.hf.co;";
// Cross-origin isolation, for multithreaded ONNX Runtime WASM.
const ISOLATION = {
  cross_origin_embedder_policy: 'require-corp',
  cross_origin_opener_policy: 'same-origin',
};
// The bench (page, fixtures) stays out. The caption workers, ONNX Runtime and the bundled Silero
// VAD ship; Whisper's weights never do: they're downloaded on demand, after the user agrees.
const FORBIDDEN = [
  /(^|\/)bench(\/|\.|-)/,
  /\.flac$/,
  /^(?!assets\/silero_vad_16k_op15-[\w-]+\.onnx$).*\.onnx$/,
];
// What captions need in the build.
const REQUIRED = [
  /^assets\/asr\.worker-/,
  /^assets\/vad\.worker-/,
  /^assets\/ort-wasm-/,
  /^assets\/silero_vad_16k_op15-/,
];

const problems = [];
const manifest = JSON.parse(await readFile(path.join(DIST, 'manifest.json'), 'utf8'));

if (JSON.stringify(manifest.permissions) !== JSON.stringify(PERMISSIONS)) {
  problems.push(
    `permissions are ${JSON.stringify(manifest.permissions)}, expected ${JSON.stringify(PERMISSIONS)}`,
  );
}
for (const key of ['host_permissions', 'optional_permissions', 'optional_host_permissions']) {
  if (key in manifest) problems.push(`manifest has ${key}: ${JSON.stringify(manifest[key])}`);
}
if (manifest.content_security_policy?.extension_pages !== CSP) {
  problems.push(
    `extension_pages CSP is ${JSON.stringify(manifest.content_security_policy?.extension_pages)}`,
  );
}
for (const [key, value] of Object.entries(ISOLATION)) {
  if (manifest[key]?.value !== value) {
    problems.push(`${key} is ${JSON.stringify(manifest[key])}, expected ${JSON.stringify(value)}`);
  }
}

const files = (await readdir(DIST, { recursive: true, withFileTypes: true }))
  .filter((entry) => entry.isFile())
  .map((entry) =>
    path.relative(DIST, path.join(entry.parentPath, entry.name)).replaceAll(path.sep, '/'),
  );
for (const file of files) {
  if (FORBIDDEN.some((pattern) => pattern.test(file))) {
    problems.push(`dev-only file in dist/: ${file}`);
  }
}

for (const pattern of REQUIRED) {
  if (!files.some((file) => pattern.test(file))) problems.push(`missing from dist/: ${pattern}`);
}

if (problems.length > 0) {
  for (const problem of problems) console.error(`✗ ${problem}`);
  process.exit(1);
}
console.log(
  `✓ dist/ (${String(files.length)} files): approved permissions, CSP and isolation, caption runtime, no dev-only files.`,
);
