#!/usr/bin/env node
/* global chrome, document -- used inside callbacks evaluated in the browser, not in Node */
// Local-only probe for the P2 capture spike; not part of CI.
//
// Real tab capture needs an action invocation, which Playwright can't click. This drives Chromium
// over CDP instead: `Extensions.triggerAction` runs the extension's action on a tab (the same path
// as a toolbar click), which opens the real popup with the activeTab grant. Everything after that
// (popup click, service worker, offscreen document, worklet) is the production code path.
//
// Usage (from the repo root, after `npm run build`, with `npm run serve:test-pages` running):
//   node docs/spikes/p2-capture-probe.mjs <path-to-chromium> [headed]
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from '@playwright/test';

const [chromePath, mode] = process.argv.slice(2);
if (!chromePath) throw new Error('usage: p2-capture-probe.mjs <path-to-chromium> [headed]');
const dist = path.resolve('dist');
const TEST_PAGE = 'http://127.0.0.1:8137/';
const PORT = 9335;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const log = (...parts) => console.log('[probe]', ...parts);

const browserProcess = spawn(
  chromePath,
  [
    `--remote-debugging-port=${String(PORT)}`,
    `--user-data-dir=${mkdtempSync(path.join(os.tmpdir(), 'tabscribe-probe-'))}`,
    `--disable-extensions-except=${dist}`,
    `--load-extension=${dist}`,
    '--enable-unsafe-extension-debugging',
    '--autoplay-policy=no-user-gesture-required',
    '--no-first-run',
    '--no-default-browser-check',
    ...(mode === 'headed' ? [] : ['--headless=new']),
    'about:blank',
  ],
  { stdio: 'ignore' },
);

/** Minimal raw CDP client: Playwright can't attach to the action popup it didn't open. */
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
  const state = () =>
    sw(async () => (await chrome.storage.session.get('captureState')).captureState);
  const offscreenCount = () =>
    sw(
      async () =>
        (await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] })).length,
    );
  const badge = () => sw(() => chrome.action.getBadgeText({}));
  const targets = async () =>
    (await cdp.send('Target.getTargets', { filter: [{}] })).result.targetInfos;

  async function openTestTab(query = '') {
    const page = await context.newPage();
    await page.goto(TEST_PAGE + query);
    await page.evaluate(() => document.querySelector('audio')?.play());
    await page.bringToFront();
    const tabId = await sw(
      async () => (await chrome.tabs.query({ active: true, lastFocusedWindow: true }))[0]?.id,
    );
    return { page, tabId };
  }

  /** Runs the extension action on the page's tab and returns an evaluator for the real popup. */
  async function openPopup(page) {
    const before = new Set((await targets()).map((t) => t.targetId));
    const tab = (await targets()).find((t) => t.type === 'tab' && t.url === page.url());
    const run = await cdp.send('Extensions.triggerAction', {
      id: extensionId,
      targetId: tab.targetId,
    });
    if (run.error) throw new Error(run.error.message);
    let popup;
    for (let i = 0; i < 40 && !popup; i++) {
      await sleep(250);
      popup = (await targets()).find(
        (t) =>
          !before.has(t.targetId) && t.type === 'page' && t.url.endsWith('/src/popup/index.html'),
      );
    }
    if (!popup) throw new Error('popup did not open');
    const { sessionId } = (
      await cdp.send('Target.attachToTarget', { targetId: popup.targetId, flatten: true })
    ).result;
    const evaluate = async (expression) =>
      (
        await cdp.send(
          'Runtime.evaluate',
          { expression, awaitPromise: true, returnByValue: true },
          sessionId,
        )
      ).result?.result?.value;
    const text = (id) => evaluate(`document.getElementById('${id}').textContent`);
    await sleep(500);
    return {
      text,
      click: (id) => evaluate(`document.getElementById('${id}').click()`),
      close: () => cdp.send('Target.closeTarget', { targetId: popup.targetId }),
      async waitForStatus(pattern) {
        for (let i = 0; i < 60; i++) {
          const status = await text('status');
          if (pattern.test(status)) return status;
          await sleep(250);
        }
        return text('status');
      },
      async levels(seconds) {
        const out = [];
        for (let i = 0; i < seconds; i++) {
          await sleep(1000);
          out.push(await text('level'));
        }
        return out.join(' | ');
      },
    };
  }

  const summary = async (label) =>
    log(
      label,
      JSON.stringify({
        state: await state(),
        offscreen: await offscreenCount(),
        badge: await badge(),
      }),
    );

  // 1. Gesture: popup URL opened as a plain tab is NOT an invocation.
  const a = await openTestTab();
  const plain = await context.newPage();
  await plain.goto(`chrome-extension://${extensionId}/src/popup/index.html`);
  log(
    '1 control reply:',
    JSON.stringify(
      await plain.evaluate(
        (id) => chrome.runtime.sendMessage({ type: 'capture/start', tabId: id }),
        a.tabId,
      ),
    ),
  );
  await plain.evaluate(() => chrome.runtime.sendMessage({ type: 'capture/stop' }));
  await plain.close();

  // 2. Gesture: popup opened by the action, Start clicked; SW calls getMediaStreamId.
  await a.page.bringToFront();
  let popup = await openPopup(a.page);
  log('2 popup before start:', await popup.text('status'));
  await popup.click('start');
  log(
    '2 status:',
    await popup.waitForStatus(/Capturing|failed/),
    '| error:',
    await popup.text('error'),
  );
  await summary('2 after start');
  log('2 levels over the tone steps:', await popup.levels(10));

  // 3. Popup closed then reopened mid-capture: meter must resume.
  await popup.close();
  await sleep(1500);
  popup = await openPopup(a.page);
  log('3 reopened status:', await popup.text('status'), '| levels:', await popup.levels(3));

  // 4. Tab muted by the user.
  await sw((id) => chrome.tabs.update(id, { muted: true }), a.tabId);
  // A full 10 s tone cycle, so muting is distinguishable from the tone's own quiet steps.
  log('4 muted levels:', await popup.levels(10));
  log(
    '4 muted: tab audible flag:',
    await sw(async (id) => (await chrome.tabs.get(id)).audible, a.tabId),
  );
  await sw((id) => chrome.tabs.update(id, { muted: false }), a.tabId);

  await popup.close();

  // 5. Same-origin in-tab navigation, then cross-origin navigation.
  await a.page.goto(TEST_PAGE + '?nav=1');
  await a.page.evaluate(() => document.querySelector('audio')?.play());
  await sleep(1500);
  await summary('5 after same-origin navigation');
  popup = await openPopup(a.page);
  log('5 levels after navigation:', await popup.levels(3));
  await popup.close();
  await a.page.goto(TEST_PAGE.replace('127.0.0.1', 'localhost') + '?nav=2');
  await a.page.evaluate(() => document.querySelector('audio')?.play());
  await sleep(1500);
  await summary('5 after cross-origin navigation');

  // 6. Start on a second tab while the first is captured.
  const b = await openTestTab('?second=1');
  popup = await openPopup(b.page);
  log(
    '6 popup on second tab:',
    await popup.text('status'),
    '| start label:',
    await popup.text('start'),
  );
  await popup.click('start');
  log('6 status:', await popup.waitForStatus(/Capturing this tab|failed/));
  const s6 = await state();
  log(
    '6 captured tab is second tab:',
    s6?.tabId === b.tabId,
    JSON.stringify(s6),
    '| offscreen:',
    await offscreenCount(),
  );
  log('6 levels:', await popup.levels(2));

  // 7. User Stop.
  await popup.click('stop');
  log('7 status after stop:', await popup.waitForStatus(/Idle/));
  await summary('7 after stop');

  await popup.close();

  // 8. Closing the captured tab tears down.
  popup = await openPopup(b.page);
  await popup.click('start');
  await popup.waitForStatus(/Capturing|failed/);
  await popup.close();
  await b.page.close();
  await sleep(1500);
  await summary('8 after closing captured tab');

  cdp.close();
  await browser.close();
}

try {
  await main();
} catch (error) {
  console.error('[probe] FAILED', error);
  process.exitCode = 1;
} finally {
  browserProcess.kill();
}
