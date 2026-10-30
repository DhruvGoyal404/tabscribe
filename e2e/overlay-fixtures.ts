// Fixtures for the caption overlay tests.
//
// Playwright can't click the extension's action, so there's no activeTab grant to inject under.
// The tests load a copy of dist/ whose manifest adds one host permission, for the local test
// server only; the shipped manifest stays activeTab-only (check:dist enforces it). An extension
// page stands in for the offscreen document: it accepts the overlay's port and sends it captions.
// It's a real page in the copy, so its listener is registered as the page loads, like the
// offscreen document's.
import { cpSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import os from 'node:os';
import path from 'node:path';
import {
  test as base,
  chromium,
  expect,
  type BrowserContext,
  type Page,
  type Worker,
} from '@playwright/test';

const dist = path.resolve(import.meta.dirname, '../dist');
const WORKER_URL = /^chrome-extension:\/\/([a-p]{32})\/service-worker-loader\.js$/;
export const OVERLAY_FILE = 'src/overlay/overlay.js';

/** The stand-in for the offscreen document, added to the test copy only. */
const SESSION_PAGE = 'e2e-session.html';
const SESSION_HTML = `<!doctype html><title>e2e session</title><script src="e2e-session.js"></script>`;
const SESSION_JS = `
  self.overlayPorts = [];
  chrome.runtime.onConnect.addListener((port) => {
    if (port.name === 'overlay') self.overlayPorts.push(port);
  });
  self.send = (message) => { for (const port of self.overlayPorts) port.postMessage(message); };
  self.closeAll = () => { for (const port of self.overlayPorts.splice(0)) port.disconnect(); };
`;

/** A strict CSP, as on many real sites: no inline script or style. */
const CSP = "default-src 'self'; script-src 'self'; style-src 'self'";

const PAGES: Record<string, string> = {
  '/': `<!doctype html><title>Video page</title>
    <link rel="stylesheet" href="/page.css">
    <div id="player"><p>Video</p></div>`,
  '/other': `<!doctype html><title>Other page</title><p>Other</p>`,
};
// Tries to hide anything it doesn't own: the overlay's host must survive this.
const PAGE_CSS = `tabscribe-overlay { display: none !important; opacity: 0 !important; }
  #player { width: 640px; height: 360px; background: #222; }`;

function testExtension(): string {
  const copy = mkdtempSync(path.join(os.tmpdir(), 'tabscribe-e2e-'));
  cpSync(dist, copy, { recursive: true });
  const manifestPath = path.join(copy, 'manifest.json');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  manifest.host_permissions = ['http://127.0.0.1/*'];
  writeFileSync(manifestPath, JSON.stringify(manifest));
  writeFileSync(path.join(copy, SESSION_PAGE), SESSION_HTML);
  writeFileSync(path.join(copy, 'e2e-session.js'), SESSION_JS);
  return copy;
}

function serve(): Promise<Server> {
  const server = createServer((request, response) => {
    if (request.url === '/page.css') {
      response.writeHead(200, { 'content-type': 'text/css' }).end(PAGE_CSS);
      return;
    }
    const html = PAGES[request.url ?? ''];
    if (html === undefined) response.writeHead(404).end();
    else {
      response
        .writeHead(200, { 'content-type': 'text/html', 'content-security-policy': CSP })
        .end(html);
    }
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      resolve(server);
    });
  });
}

/** The stand-in for the offscreen document, in an extension page. */
export interface FakeSession {
  /** Resolves once the overlay has connected (or `count` overlays have). */
  connected(count?: number): Promise<void>;
  send(message: unknown): Promise<void>;
  /** Closes every overlay port, as the offscreen document closing does. */
  close(): Promise<void>;
}

interface Fixtures {
  context: BrowserContext;
  worker: Worker;
  origin: string;
  page: Page;
  session: FakeSession;
  inject: () => Promise<void>;
  overlay: () => Promise<OverlayView | null>;
}

/** What's inside the overlay's closed shadow root, read through the DevTools protocol. */
export interface OverlayView {
  shadowRootType: string;
  hidden: boolean;
  final: string;
  partial: string;
}

interface DomNode {
  nodeType: number;
  localName?: string;
  nodeValue?: string;
  attributes?: string[];
  children?: DomNode[];
  shadowRoots?: (DomNode & { shadowRootType?: string })[];
}

const text = (node: DomNode | undefined): string =>
  node?.children
    ?.map((child) => (child.nodeType === 3 ? (child.nodeValue ?? '') : text(child)))
    .join('') ?? '';

function find(node: DomNode, test: (node: DomNode) => boolean): DomNode | undefined {
  if (test(node)) return node;
  for (const child of [...(node.children ?? []), ...(node.shadowRoots ?? [])]) {
    const found = find(child, test);
    if (found) return found;
  }
  return undefined;
}

export const test = base.extend<Fixtures>({
  // eslint-disable-next-line no-empty-pattern -- Playwright fixtures must destructure their deps.
  context: async ({}, use) => {
    const extension = testExtension();
    const context = await chromium.launchPersistentContext('', {
      channel: 'chromium',
      args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
    });
    await use(context);
    await context.close();
  },
  worker: async ({ context }, use) => {
    const find = () => context.serviceWorkers().find((w) => WORKER_URL.test(w.url()));
    await expect.poll(find).toBeDefined();
    const worker = find();
    if (!worker) throw new Error('Extension service worker not found');
    await use(worker);
  },
  // eslint-disable-next-line no-empty-pattern -- Playwright fixtures must destructure their deps.
  origin: async ({}, use) => {
    const server = await serve();
    await use(`http://127.0.0.1:${String((server.address() as AddressInfo).port)}`);
    server.close();
  },
  page: async ({ context, origin }, use) => {
    const page = await context.newPage();
    await page.goto(`${origin}/`);
    await use(page);
  },
  session: async ({ context, worker }, use) => {
    const extensionId = new URL(worker.url()).host;
    const host = await context.newPage();
    await host.goto(`chrome-extension://${extensionId}/${SESSION_PAGE}`);
    await use({
      connected: (count = 1) =>
        expect
          .poll(() => host.evaluate<number>('self.overlayPorts.length'))
          .toBeGreaterThanOrEqual(count),
      // Strings, as e2e/ has no DOM or chrome types.
      send: (message) => host.evaluate(`self.send(${JSON.stringify(message)})`),
      close: () => host.evaluate(`self.closeAll()`),
    });
  },
  inject: async ({ worker, page }, use) => {
    await use(async () => {
      await worker.evaluate(`(async () => {
        const [tab] = await chrome.tabs.query({ url: ${JSON.stringify(page.url())} });
        await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['${OVERLAY_FILE}'] });
      })()`);
    });
  },
  overlay: async ({ page }, use) => {
    const cdp = await page.context().newCDPSession(page);
    await use(async () => {
      const { root } = (await cdp.send('DOM.getDocument', { depth: -1, pierce: true })) as {
        root: DomNode;
      };
      const host = find(root, (node) => node.localName === 'tabscribe-overlay');
      const shadow = host?.shadowRoots?.[0];
      if (!shadow) return null;
      const box = find(shadow, (node) => node.attributes?.includes('captions') ?? false);
      const spans = box?.children?.filter((child) => child.localName === 'span') ?? [];
      return {
        shadowRootType: shadow.shadowRootType ?? '',
        hidden: box?.attributes?.includes('hidden') ?? true,
        final: text(spans[0]),
        partial: text(spans[1]),
      };
    });
  },
});

export { expect };
