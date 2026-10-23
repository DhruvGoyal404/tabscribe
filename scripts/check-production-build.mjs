#!/usr/bin/env node
// Checks the production build in dist/ after `npm run build`:
// - the manifest keeps exactly the approved permissions, no host permissions, and the approved CSP;
// - nothing from the dev-only bench (page, fixtures, ONNX Runtime, model files) leaked in.
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DIST = fileURLToPath(new URL('../dist/', import.meta.url));
const PERMISSIONS = ['activeTab', 'tabCapture', 'offscreen', 'scripting', 'storage'];
const CSP =
  "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'; " +
  "connect-src 'self' https://huggingface.co https://*.hf.co;";
// Until P3b wires ASR into the offscreen document, its workers and models are bench-only.
const FORBIDDEN = [
  /(^|\/)bench(\/|\.|-)/,
  /\.flac$/,
  /\.onnx$/,
  /ort-wasm/,
  /asr\.worker/,
  /vad\.worker/,
  /silero/,
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

if (problems.length > 0) {
  for (const problem of problems) console.error(`✗ ${problem}`);
  process.exit(1);
}
console.log(
  `✓ dist/ (${String(files.length)} files): approved permissions and CSP, no dev-only files.`,
);
