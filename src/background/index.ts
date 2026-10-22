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
  type Message,
  type Reply,
} from '../shared/messages';
import { mutedChange } from './tab-sync';

const OFFSCREEN_PATH = 'src/offscreen/index.html';
const BADGE_COLOR = '#1f5ed6';

async function readState(): Promise<CaptureState> {
  const stored: unknown = (await chrome.storage.session.get(STATE_KEY))[STATE_KEY];
  return isCaptureState(stored) ? stored : IDLE;
}

async function dispatch(event: CaptureEvent): Promise<CaptureState> {
  const next = transition(await readState(), event);
  await chrome.storage.session.set({ [STATE_KEY]: next });
  if (next.status === 'capturing') {
    await chrome.action.setBadgeBackgroundColor({ color: BADGE_COLOR });
    await chrome.action.setBadgeText({ text: 'ON' });
  } else {
    await chrome.action.setBadgeText({ text: '' });
  }
  return next;
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
      reasons: [chrome.offscreen.Reason.USER_MEDIA],
      justification: 'Plays captured tab audio back to the user and measures its level.',
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
  try {
    if (await hasOffscreenDocument()) {
      // Stop tracks and close the AudioContext explicitly before the document goes away.
      await sendToOffscreen({ type: 'offscreen/stop' }).catch((error: unknown) => {
        console.warn('[tabscribe] offscreen stop failed', error);
      });
      await chrome.offscreen.closeDocument();
    }
  } finally {
    await chrome.action.setBadgeText({ text: '' });
  }
}

// --- Capture lifecycle --------------------------------------------------------------------------

async function startCapture(tabId: number): Promise<void> {
  const current = await readState();
  if (current.status === 'capturing' && current.tabId === tabId) return;
  if (current.status !== 'idle' && current.status !== 'error') await stopCapture();

  await dispatch({ type: 'start', tabId });
  try {
    // Allowed because opening the popup invoked the extension on this tab (an activeTab grant).
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
    // mutedInfo isn't one of the fields gated by the "tabs" permission. Later changes arrive via
    // tabs.onUpdated, queued behind this start.
    const { mutedInfo } = await chrome.tabs.get(tabId);
    await ensureOffscreenDocument();
    await sendToOffscreen({
      type: 'offscreen/start',
      streamId,
      muted: mutedInfo?.muted ?? false,
    });
    await dispatch({ type: 'started' });
  } catch (error) {
    await fail(error);
    throw error;
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
        void settle(serialize(() => startCapture(message.tabId))).then(sendResponse);
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
  if (!change.mutedInfo) return;
  void serialize(async () => {
    const muted = mutedChange(await readState(), tabId, change);
    if (muted !== undefined) await sendToOffscreen({ type: 'offscreen/set-muted', muted });
  }).catch((error: unknown) => {
    console.warn('[tabscribe] could not follow the tab mute state', error);
  });
});

chrome.runtime.onInstalled.addListener(({ reason }) => {
  console.info(`[tabscribe] service worker ready (${reason})`);
});
