import {
  CAPTION_MODEL,
  describePlan,
  planBackend,
  planBytes,
  type CaptionPlan,
} from '../asr/backend';
import { gpuInfo } from '../asr/gpu';
import { isCached } from '../asr/models';
import { IDLE, isCaptureState, STATE_KEY, type CaptureState } from '../shared/capture-state';
import { describeStatus, isCaptionMessage, type Caption } from '../shared/captions';
import { MIN_DBFS } from '../shared/level';
import {
  isLevelMessage,
  isReply,
  SESSION_PORT,
  WEBGPU_FAILURE_KEY,
  type Message,
} from '../shared/messages';

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
  captionStatus: element('caption-status', HTMLParagraphElement),
  caption: element('caption', HTMLParagraphElement),
  final: element('final', HTMLSpanElement),
  partial: element('partial', HTMLSpanElement),
  plan: element('plan', HTMLParagraphElement),
  consent: element('consent', HTMLParagraphElement),
  start: element('start', HTMLButtonElement),
  stop: element('stop', HTMLButtonElement),
  error: element('error', HTMLParagraphElement),
  version: element('version', HTMLParagraphElement),
};

let state: CaptureState = IDLE;
let activeTabId: number | undefined;
let sessionPort: chrome.runtime.Port | null = null;

/** How captions would run if started now, and whether that needs a download first. */
let planned: { plan: CaptionPlan; cached: boolean } | null = null;

const mb = (bytes: number): string => `${(bytes / 1e6).toFixed(1)} MB`;
const capitalise = (text: string): string => (text[0]?.toUpperCase() ?? '') + text.slice(1);

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
  // The button names the download, so clicking it is the agreement to that download.
  const download = planned && !planned.cached ? `Download ${mb(planBytes(planned.plan))} and ` : '';
  const action = state.status === 'capturing' && !capturingThisTab ? 'capture this tab' : 'start';
  ui.start.textContent = download ? download + action : capitalise(action);
  ui.start.disabled = activeTabId === undefined || busy || capturingThisTab || !planned;
  ui.stop.textContent = state.status === 'error' ? 'Dismiss' : 'Stop';
  ui.stop.disabled = !(
    state.status === 'starting' ||
    state.status === 'capturing' ||
    state.status === 'error'
  );
  ui.error.hidden = state.status !== 'error';
  ui.error.textContent = state.status === 'error' ? state.message : '';

  if (planned) {
    ui.plan.textContent = `Captions: ${describePlan(planned.plan)}. ${capitalise(planned.plan.reason)}.`;
    ui.consent.hidden = planned.cached;
    ui.consent.textContent =
      `Captions need a one-time ${mb(planBytes(planned.plan))} download of the ${CAPTION_MODEL} ` +
      'speech model from huggingface.co. It stays on this device, and speech never leaves it.';
  }

  syncSessionPort();
  if (state.status !== 'capturing') {
    showLevel(MIN_DBFS, false);
    clearCaptions();
  }
}

function showLevel(rmsDbfs: number, clipped: boolean): void {
  ui.meter.value = Math.max(METER_FLOOR_DBFS, Math.min(0, rmsDbfs));
  ui.level.textContent = rmsDbfs <= METER_FLOOR_DBFS ? '– dBFS' : `${rmsDbfs.toFixed(1)} dBFS`;
  ui.level.classList.toggle('clipped', clipped);
}

function showCaption(caption: Caption): void {
  ui.caption.hidden = false;
  if (caption.kind === 'final') {
    // The newest final, and nothing pending: piece 4 adds a scrolling transcript.
    if (caption.text) ui.final.textContent = caption.text;
    ui.partial.textContent = '';
  } else ui.partial.textContent = caption.text;
}

function clearCaptions(): void {
  ui.captionStatus.hidden = true;
  ui.captionStatus.textContent = '';
  ui.caption.hidden = true;
  ui.final.textContent = '';
  ui.partial.textContent = '';
}

// Levels, caption status and captions stream straight from the offscreen document. Connect
// whenever a capture is running, so reopening the popup mid-capture picks them back up.
function syncSessionPort(): void {
  if (state.status === 'capturing' && !sessionPort) {
    const port = chrome.runtime.connect({ name: SESSION_PORT });
    port.onMessage.addListener((message: unknown) => {
      if (isLevelMessage(message)) showLevel(message.rmsDbfs, message.clipped);
      else if (isCaptionMessage(message)) {
        if (message.type === 'captions/status') {
          ui.captionStatus.hidden = false;
          ui.captionStatus.textContent = describeStatus(message.status);
        } else showCaption(message.caption);
      }
    });
    port.onDisconnect.addListener(() => {
      // Reading lastError marks it handled (e.g. the offscreen document has already closed).
      if (chrome.runtime.lastError) console.debug('[tabscribe] session port closed');
      if (sessionPort === port) sessionPort = null;
      showLevel(MIN_DBFS, false);
    });
    sessionPort = port;
  } else if (state.status !== 'capturing' && sessionPort) {
    sessionPort.disconnect();
    sessionPort = null;
  }
}

/** Plans the caption backend here: the offscreen document runs on the same GPU. */
async function refreshPlan(): Promise<void> {
  const stored: unknown = (await chrome.storage.local.get(WEBGPU_FAILURE_KEY))[WEBGPU_FAILURE_KEY];
  const plan = planBackend({
    gpu: await gpuInfo(),
    webgpuFailure: typeof stored === 'string' ? stored : null,
    crossOriginIsolated: self.crossOriginIsolated,
    hardwareConcurrency: navigator.hardwareConcurrency,
  });
  planned = { plan, cached: await isCached(CAPTION_MODEL, plan.dtype) };
  render();
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
  const current = planned;
  if (!current) return;
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
  // The button said "Download … and start" exactly when the model wasn't cached.
  await send({
    type: 'capture/start',
    tabId,
    streamId,
    plan: current.plan,
    allowDownload: !current.cached,
  });
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
  // A finished capture may have downloaded the model.
  if (state.status === 'idle') void refreshPlan();
});

chrome.storage.local.onChanged.addListener((changes) => {
  if (WEBGPU_FAILURE_KEY in changes) void refreshPlan();
});

async function init(): Promise<void> {
  ui.version.textContent = `v${chrome.runtime.getManifest().version}`;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  activeTabId = tab?.id;
  const stored: unknown = (await chrome.storage.session.get(STATE_KEY))[STATE_KEY];
  state = isCaptureState(stored) ? stored : IDLE;
  render();
  await refreshPlan();
}

void init();
