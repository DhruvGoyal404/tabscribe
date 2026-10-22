import path from 'node:path';
import { test as base, chromium, expect, type BrowserContext } from '@playwright/test';

// Requires `npm run build` first; loads dist/ as an unpacked extension.
const extensionPath = path.resolve(import.meta.dirname, '../dist');
const WORKER_URL = /^chrome-extension:\/\/([a-p]{32})\/service-worker-loader\.js$/;

export const test = base.extend<{ context: BrowserContext; extensionId: string }>({
  // eslint-disable-next-line no-empty-pattern -- Playwright fixtures must destructure their deps.
  context: async ({}, use) => {
    // The `chromium` channel runs extensions in new headless mode.
    const context = await chromium.launchPersistentContext('', {
      channel: 'chromium',
      args: [`--disable-extensions-except=${extensionPath}`, `--load-extension=${extensionPath}`],
    });
    await use(context);
    await context.close();
  },
  extensionId: async ({ context }, use) => {
    // Poll rather than waitForEvent: the worker can register before the listener attaches.
    const workerUrl = () =>
      context
        .serviceWorkers()
        .find((w) => WORKER_URL.test(w.url()))
        ?.url();
    await expect.poll(workerUrl).toBeDefined();
    const match = WORKER_URL.exec(workerUrl() ?? '');
    if (!match?.[1]) throw new Error('Extension service worker not found');
    await use(match[1]);
  },
});

export { expect };
