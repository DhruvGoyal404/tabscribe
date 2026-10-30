#!/usr/bin/env node
/* global chrome, document -- used inside callbacks evaluated in the browser, not in Node */
// Local-only probe for P3b piece 3: real captions, end to end; not part of CI.
//
// Serves a page that plays the English bench fixture, runs the extension's action on it over CDP
// (as p2-capture-probe.mjs does, so the popup gets the activeTab grant), clicks Start and records
// the popup's plan, consent, caption status and every caption it shows. It downloads the speech
// model (the click on "Download … and start" is the consent), so run it on purpose.
//
// Usage (from the repo root, after `npm run build`):
//   node docs/spikes/p3b-captions-probe.mjs <path-to-chromium> [webgpu|wasm] [profileDir]
// `wasm` records a WebGPU failure first, so the popup plans WASM. Reuse a profileDir to start from
// cache. The window stays headed: WebGPU and timers are throttled in background windows.
import { spawn } from 'node:child_process';
import { readFileSync, mkdtempSync } from 'node:fs';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { chromium } from '@playwright/test';

const [chromePath, backend = 'webgpu', profileDir] = process.argv.slice(2);
if (!chromePath)
  throw new Error('usage: p3b-captions-probe.mjs <path-to-chromium> [webgpu|wasm] [profileDir]');
const dist = path.resolve('dist');
const FIXTURE = path.resolve('src/bench/fixtures/en-librispeech.flac');
const PORT = 9343;
const PAGE_PORT = 8139;
/** The fixture is 47.7 s; leave time for the last final. */
const LISTEN_SECONDS = 70;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const log = (...parts) => console.log('[captions]', ...parts);

const page = `<!doctype html><title>Speech</title><audio id="speech" src="/speech.flac"></audio>`;
const server = createServer((request, response) => {
  if (request.url === '/speech.flac') {
    response.writeHead(200, { 'content-type': 'audio/flac' }).end(readFileSync(FIXTURE));
  } else response.writeHead(200, { 'content-type': 'text/html' }).end(page);
}).listen(PAGE_PORT, '127.0.0.1');

const browserProcess = spawn(
  chromePath,
  [
    `--remote-debugging-port=${String(PORT)}`,
    `--user-data-dir=${profileDir ?? mkdtempSync(path.join(os.tmpdir(), 'tabscribe-captions-'))}`,
    `--disable-extensions-except=${dist}`,
    `--load-extension=${dist}`,
    '--enable-unsafe-extension-debugging',
    '--autoplay-policy=no-user-gesture-required',
    '--no-first-run',
    '--no-default-browser-check',
    'about:blank',
  ],
  { stdio: 'ignore' },
);

async function rawCdp() {
  const { webSocketDebuggerUrl } = await (
    await fetch(`http://127.0.0.1:${String(PORT)}/json/version`)
  ).json();
  const ws = new WebSocket(webSocketDebuggerUrl);
  await new Promise((resolve) => ws.addEventListener('open', resolve, { once: true }));
  let nextId = 1;
  const pending = new Map();
  ws.addEventListener('message', (event) => {
    const message = JSON.parse(String(event.data));
    pending.get(message.id)?.(message);
    pending.delete(message.id);
  });
  const send = (method, params = {}, sessionId) =>
    new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  return { send, close: () => ws.close() };
}

async function main() {
  for (let i = 0; i < 80; i++) {
    try {
      await fetch(`http://127.0.0.1:${String(PORT)}/json/version`);
      break;
    } catch {
      await sleep(250);
    }
  }
  const browser = await chromium.connectOverCDP(`http://127.0.0.1:${String(PORT)}`);
  const context = browser.contexts()[0];
  const cdp = await rawCdp();
  let worker;
  for (let i = 0; i < 40 && !worker; i++) {
    worker = context.serviceWorkers().find((w) => w.url().endsWith('/service-worker-loader.js'));
    if (!worker) await sleep(250);
  }
  const extensionId = new URL(worker.url()).host;
  const sw = (fn, arg) => worker.evaluate(fn, arg);
  if (backend === 'wasm') {
    await sw(() => chrome.storage.local.set({ webgpuFailure: 'probe: forced WASM' }));
  }
  const targets = async () =>
    (await cdp.send('Target.getTargets', { filter: [{}] })).result.targetInfos;

  const tab = await context.newPage();
  await tab.goto(`http://127.0.0.1:${String(PAGE_PORT)}/`);
  await tab.bringToFront();

  const before = new Set((await targets()).map((t) => t.targetId));
  const tabTarget = (await targets()).find((t) => t.type === 'tab' && t.url === tab.url());
  const run = await cdp.send('Extensions.triggerAction', {
    id: extensionId,
    targetId: tabTarget.targetId,
  });
  if (run.error) throw new Error(run.error.message);
  let popup;
  // A loaded machine can take a while to show the popup.
  for (let i = 0; i < 160 && !popup; i++) {
    await sleep(250);
    popup = (await targets()).find(
      (t) => !before.has(t.targetId) && t.type === 'page' && t.url.endsWith('/src/popup/index.html'),
    );
  }
  if (!popup) {
    const added = (await targets()).filter((t) => !before.has(t.targetId));
    throw new Error(`popup did not open; new targets: ${JSON.stringify(added.map((t) => [t.type, t.url]))}; triggerAction: ${JSON.stringify(run)}`);
  }
  const { sessionId } = (
    await cdp.send('Target.attachToTarget', { targetId: popup.targetId, flatten: true })
  ).result;
  const evaluate = async (expression) =>
    (await cdp.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId))
      .result?.result?.value;
  const text = (id) => evaluate(`document.getElementById('${id}').textContent`);
  for (let i = 0; i < 240 && /Checking/.test(await text('plan')); i++) await sleep(250);

  log('plan:', await text('plan'));
  log('consent:', (await evaluate(`document.getElementById('consent').hidden`)) ? '(hidden: cached)' : await text('consent'));
  log('start button:', await text('start'));
  const clicked = Date.now();
  await evaluate(`document.getElementById('start').click()`);

  // Play the speech once the model is listening, so caption timing starts at the audio.
  let status = '';
  let lastLoading = '';
  for (let i = 0; i < 1200 && !/Listening|stopped/.test(status); i++) {
    await sleep(500);
    status = await text('caption-status');
    if (/Downloading/.test(status) && status !== lastLoading && i % 10 === 0) log('  ', status);
    lastLoading = status;
  }
  log(`status after ${((Date.now() - clicked) / 1000).toFixed(1)} s:`, status);
  if (!/Listening/.test(status)) throw new Error(`captions did not start: ${status}`);
  // Once the model is ready the download offer goes, without reopening the popup.
  const consentHidden = await evaluate(`document.getElementById('consent').hidden`);
  log('while listening: consent', consentHidden ? 'hidden' : 'STILL SHOWN', '| start button:', await text('start'));
  const played = Date.now();
  await tab.evaluate(() => document.getElementById('speech').play());

  const finals = [];
  let partials = 0;
  let lastFinal = '';
  let lastPartial = '';
  for (let i = 0; i < LISTEN_SECONDS * 4; i++) {
    await sleep(250);
    const final = await text('final');
    const partial = await text('partial');
    const at = ((Date.now() - played) / 1000).toFixed(1);
    if (final && final !== lastFinal) {
      finals.push(final);
      log(`  ${at} s final: ${final}`);
      lastFinal = final;
    }
    if (partial && partial !== lastPartial) partials += 1;
    lastPartial = partial;
  }
  log('caption status at the end:', await text('caption-status'));
  log(`finals: ${String(finals.length)}, partial updates seen: ${String(partials)}`);
  log('transcript:', finals.join(' '));

  await evaluate(`document.getElementById('stop').click()`);
  await sleep(2000);
  const contexts = await sw(
    async () => (await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })).length,
  );
  log('after Stop, offscreen documents:', contexts, '| state:', JSON.stringify(await sw(async () => (await chrome.storage.session.get('captureState')).captureState)));
  log('webgpuFailure stored:', JSON.stringify(await sw(async () => (await chrome.storage.local.get('webgpuFailure')).webgpuFailure ?? null)));
  cdp.close();
  await browser.close();
} // prettier-ignore

main()
  .catch((error) => {
    console.error('[captions] FAILED', error);
    process.exitCode = 1;
  })
  .finally(() => {
    browserProcess.kill();
    server.close();
  });
