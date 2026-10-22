import path from 'node:path';
import { chromium, expect, test } from '@playwright/test';

// Requires `npm run build` first; loads dist/ as an unpacked extension.
const extensionPath = path.resolve(import.meta.dirname, '../dist');

test('extension service worker starts', async () => {
  const context = await chromium.launchPersistentContext('', {
    channel: 'chromium',
    args: [`--disable-extensions-except=${extensionPath}`, `--load-extension=${extensionPath}`],
  });
  try {
    // Poll rather than waitForEvent: the worker can register before the listener attaches.
    await expect
      .poll(() => context.serviceWorkers().map((worker) => worker.url()))
      .toContainEqual(
        expect.stringMatching(/^chrome-extension:\/\/[a-p]{32}\/service-worker-loader\.js$/),
      );
  } finally {
    await context.close();
  }
});
