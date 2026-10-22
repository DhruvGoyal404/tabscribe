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
    const worker = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
    expect(worker.url()).toMatch(/^chrome-extension:\/\/[a-p]{32}\//);
  } finally {
    await context.close();
  }
});
