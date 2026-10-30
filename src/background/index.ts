// Service worker: owns the capture state machine and the offscreen document's lifecycle.
// State lives in chrome.storage.session, not module variables, because the worker can be
// terminated and restarted at any time.
import {
  capturedTabId,
  IDLE,
  isCaptureState,
  STATE_KEY,
  transition,
  type CaptureEvent,
  type CaptureState,
} from '../shared/capture-state';
import {
  describeError,
  isMessage,
  isReply,
  settle,
  WEBGPU_FAILURE_KEY,
  type CaptionRequest,
  type Message,
  type Reply,
} from '../shared/messages';
import { badgeRefresh, badgeUpdates, mutedChange } from './tab-sync';
// One self-contained file (IIFE): nothing to load from the extension, so no web-accessible files.
import overlayScript from '../overlay/overlay.ts?iife';

const OFFSCREEN_PATH = 'src/offscreen/index.html';
const BADGE_COLOR = '#1f5ed6';

async function readState(): Promise<CaptureState> {
  const stored: unknown = (await chrome.storage.session.get(STATE_KEY))[STATE_KEY];
  return isCaptureState(stored) ? stored : IDLE;
}

async function dispatch(event: CaptureEvent): Promise<CaptureState> {
  const previous = await readState();
  const next = transition(previous, event);
  await chrome.storage.session.set({ [STATE_KEY]: next });
  for (const { tabId, text } of badgeUpdates(previous, next)) await setBadge(tabId, text);
  return next;
}

/** Per-tab badge. Chrome resets it when the tab closes, so a closed tab's failure is ignored. */
async function setBadge(tabId: number, text: string): Promise<void> {
  try {
    if (text) await chrome.action.setBadgeBackgroundColor({ tabId, color: BADGE_COLOR });
    await chrome.action.setBadgeText({ tabId, text });
  } catch (error) {
    console.debug('[tabscribe] badge not updated for tab', tabId, error);
  }
}

// Run capture operations one at a time so double clicks and overlapping triggers can't interleave.
let queue: Promise<unknown> = Promise.resolve();
function serialize<T>(task: () => Promise<T>): Promise<T> {
  const run = queue.then(task);
  queue = run.catch(() => undefined);
  return run;
}

// --- Offscreen document -------------------------------------------------------------------------

let creatingOffscreen: Promise<void> | null = null;

async function hasOffscreenDocument(): Promise<boolean> {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_PATH)],
  });
  return contexts.length > 0;
}

async function ensureOffscreenDocument(): Promise<void> {
  if (await hasOffscreenDocument()) return;
  // Share one in-flight creation: only one offscreen document may exist per extension.
  creatingOffscreen ??= chrome.offscreen
    .createDocument({
      url: OFFSCREEN_PATH,
      reasons: [chrome.offscreen.Reason.USER_MEDIA, chrome.offscreen.Reason.WORKERS],
      justification:
        'Plays captured tab audio back to the user, measures its level, and captions it on the ' +
        'device in voice-detection and speech-recognition workers.',
    })
    .finally(() => {
      creatingOffscreen = null;
    });
  await creatingOffscreen;
}

async function sendToOffscreen(message: Message): Promise<void> {
  const reply: unknown = await chrome.runtime.sendMessage(message);
  if (!isReply(reply)) throw new Error('The offscreen document did not reply.');
  if (!reply.ok) throw new Error(reply.error);
}

async function teardown(): Promise<void> {
  if (!(await hasOffscreenDocument())) return;
  // Stop tracks and close the AudioContext explicitly before the document goes away.
  await sendToOffscreen({ type: 'offscreen/stop' }).catch((error: unknown) => {
    console.warn('[tabscribe] offscreen stop failed', error);
  });
  await chrome.offscreen.closeDocument();
}

// --- Capture lifecycle --------------------------------------------------------------------------

async function startCapture(
  tabId: number,
  streamId: string,
  captions: CaptionRequest,
): Promise<void> {
  const current = await readState();
  if (current.status === 'capturing' && current.tabId === tabId) return;
  if (current.status !== 'idle' && current.status !== 'error') await stopCapture();

  await dispatch({ type: 'start', tabId });
  try {
    // mutedInfo isn't one of the fields gated by the "tabs" permission. Later changes arrive via
    // tabs.onUpdated, queued behind this start.
    const { mutedInfo } = await chrome.tabs.get(tabId);
    await ensureOffscreenDocument();
    await sendToOffscreen({
      type: 'offscreen/start',
      streamId,
      muted: mutedInfo?.muted ?? false,
      plan: captions.plan,
      allowDownload: captions.allowDownload,
    });
    await dispatch({ type: 'started' });
  } catch (error) {
    await fail(error);
    throw error;
  }
  // After the offscreen document is up, so the overlay's port has somewhere to connect.
  await showOverlay(tabId);
}

/**
 * Injects the caption overlay into the captured tab, under the activeTab grant from the popup
 * click. Captions go on in the popup if this fails (a page scripts can't run in, for example).
 */
async function showOverlay(tabId: number): Promise<void> {
  try {
    await chrome.scripting.executeScript({ target: { tabId }, files: [overlayScript] });
  } catch (error) {
    console.warn('[tabscribe] no caption overlay on this page:', describeError(error));
  }
}

async function stopCapture(): Promise<void> {
  const state = await dispatch({ type: 'stop' });
  try {
    await teardown();
  } finally {
    if (state.status === 'stopping') await dispatch({ type: 'stopped' });
  }
}

async function fail(error: unknown): Promise<void> {
  const message = describeError(error);
  console.error('[tabscribe] capture failed:', message);
  await dispatch({ type: 'failed', message });
  await teardown().catch((teardownError: unknown) => {
    console.error('[tabscribe] teardown after failure failed', teardownError);
  });
}

// --- Event wiring -------------------------------------------------------------------------------

chrome.runtime.onMessage.addListener(
  (message: unknown, sender, sendResponse: (reply: Reply) => void) => {
    if (sender.id !== chrome.runtime.id || !isMessage(message)) return false;
    switch (message.type) {
      case 'capture/start':
        void settle(
          serialize(() =>
            startCapture(message.tabId, message.streamId, {
              plan: message.plan,
              allowDownload: message.allowDownload,
            }),
          ),
        ).then(sendResponse);
        return true;
      case 'capture/stop':
        void settle(serialize(stopCapture)).then(sendResponse);
        return true;
      case 'offscreen/track-ended':
        void serialize(stopCapture);
        return false;
      case 'offscreen/error':
        void serialize(() => fail(message.message));
        return false;
      case 'offscreen/webgpu-failed':
        // The popup plans WASM from now on. Captions stop with an error; capture goes on.
        void chrome.storage.local.set({ [WEBGPU_FAILURE_KEY]: message.message });
        return false;
      default:
        // offscreen/start, offscreen/set-muted and offscreen/stop are for the offscreen document.
        return false;
    }
  },
);

chrome.tabs.onRemoved.addListener((tabId) => {
  void serialize(async () => {
    if (capturedTabId(await readState()) === tabId) await stopCapture();
  });
});

chrome.tabs.onUpdated.addListener((tabId, change) => {
  if (!change.mutedInfo && change.status === undefined) return;
  void serialize(async () => {
    const state = await readState();
    const badge = badgeRefresh(state, tabId, change);
    if (badge) await setBadge(badge.tabId, badge.text);
    const muted = mutedChange(state, tabId, change);
    if (muted !== undefined) await sendToOffscreen({ type: 'offscreen/set-muted', muted });
  }).catch((error: unknown) => {
    console.warn('[tabscribe] could not sync the captured tab', error);
  });
});

chrome.runtime.onInstalled.addListener(({ reason }) => {
  console.info(`[tabscribe] service worker ready (${reason})`);
  // A new version may have fixed what made WebGPU fail: plan it again.
  if (reason === 'update') void chrome.storage.local.remove(WEBGPU_FAILURE_KEY);
});
