import { IDLE, isCaptureState, STATE_KEY, type CaptureState } from '../shared/capture-state';
import { MIN_DBFS } from '../shared/level';
import { isLevelMessage, isReply, LEVELS_PORT, type Message } from '../shared/messages';

const METER_FLOOR_DBFS = -60;

function element<T extends HTMLElement>(id: string, type: new () => T): T {
  const found = document.getElementById(id);
  if (!(found instanceof type)) throw new Error(`Popup is missing #${id}`);
  return found;
}

const ui = {
  status: element('status', HTMLParagraphElement),
  meter: element('meter', HTMLMeterElement),
  level: element('level', HTMLSpanElement),
  start: element('start', HTMLButtonElement),
  stop: element('stop', HTMLButtonElement),
  error: element('error', HTMLParagraphElement),
  version: element('version', HTMLParagraphElement),
};

let state: CaptureState = IDLE;
let activeTabId: number | undefined;
let levelPort: chrome.runtime.Port | null = null;

function describe(current: CaptureState): string {
  switch (current.status) {
    case 'idle':
      return 'Idle';
    case 'starting':
      return 'Starting…';
    case 'capturing':
      return current.tabId === activeTabId ? 'Capturing this tab' : 'Capturing another tab';
    case 'stopping':
      return 'Stopping…';
    case 'error':
      return 'Capture failed';
  }
}

function render(): void {
  const capturingThisTab = state.status === 'capturing' && state.tabId === activeTabId;
  const busy = state.status === 'starting' || state.status === 'stopping';

  ui.status.textContent = describe(state);
  ui.start.textContent =
    state.status === 'capturing' && !capturingThisTab ? 'Capture this tab' : 'Start';
  ui.start.disabled = activeTabId === undefined || busy || capturingThisTab;
  ui.stop.textContent = state.status === 'error' ? 'Dismiss' : 'Stop';
  ui.stop.disabled = !(
    state.status === 'starting' ||
    state.status === 'capturing' ||
    state.status === 'error'
  );
  ui.error.hidden = state.status !== 'error';
  ui.error.textContent = state.status === 'error' ? state.message : '';

  syncLevelPort();
  if (state.status !== 'capturing') showLevel(MIN_DBFS, false);
}

function showLevel(rmsDbfs: number, clipped: boolean): void {
  ui.meter.value = Math.max(METER_FLOOR_DBFS, Math.min(0, rmsDbfs));
  ui.level.textContent = rmsDbfs <= METER_FLOOR_DBFS ? '– dBFS' : `${rmsDbfs.toFixed(1)} dBFS`;
  ui.level.classList.toggle('clipped', clipped);
}

// Levels stream straight from the offscreen document. Connect whenever a capture is running,
// so reopening the popup mid-capture picks the meter back up.
function syncLevelPort(): void {
  if (state.status === 'capturing' && !levelPort) {
    const port = chrome.runtime.connect({ name: LEVELS_PORT });
    port.onMessage.addListener((message: unknown) => {
      if (isLevelMessage(message)) showLevel(message.rmsDbfs, message.clipped);
    });
    port.onDisconnect.addListener(() => {
      // Reading lastError marks it handled (e.g. the offscreen document has already closed).
      if (chrome.runtime.lastError) console.debug('[tabscribe] level port closed');
      if (levelPort === port) levelPort = null;
      showLevel(MIN_DBFS, false);
    });
    levelPort = port;
  } else if (state.status !== 'capturing' && levelPort) {
    levelPort.disconnect();
    levelPort = null;
  }
}

async function send(message: Message): Promise<void> {
  try {
    const reply: unknown = await chrome.runtime.sendMessage(message);
    // Failures also land in the shared state, which render() shows; this covers a missing reply.
    if (!isReply(reply)) throw new Error('No reply from the service worker.');
  } catch (error) {
    ui.error.hidden = false;
    ui.error.textContent = error instanceof Error ? error.message : String(error);
  }
}

async function start(tabId: number): Promise<void> {
  let streamId: string;
  try {
    // Allowed because opening the popup invoked the extension on this tab (an activeTab grant).
    // Requested here, not in the service worker: the id only works in a context that is
    // cross-origin isolated like the offscreen document that consumes it.
    streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });
  } catch (error) {
    ui.error.hidden = false;
    ui.error.textContent = error instanceof Error ? error.message : String(error);
    return;
  }
  await send({ type: 'capture/start', tabId, streamId });
}

ui.start.addEventListener('click', () => {
  if (activeTabId !== undefined) void start(activeTabId);
});
ui.stop.addEventListener('click', () => {
  void send({ type: 'capture/stop' });
});

chrome.storage.session.onChanged.addListener((changes) => {
  const change = changes[STATE_KEY];
  if (!change) return;
  state = isCaptureState(change.newValue) ? change.newValue : IDLE;
  render();
});

async function init(): Promise<void> {
  ui.version.textContent = `v${chrome.runtime.getManifest().version}`;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  activeTabId = tab?.id;
  const stored: unknown = (await chrome.storage.session.get(STATE_KEY))[STATE_KEY];
  state = isCaptureState(stored) ? stored : IDLE;
  render();
}

void init();
