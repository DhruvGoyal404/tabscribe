// Offscreen document (reason USER_MEDIA). Turns a tab stream id into audio: plays it back so the
// tab stays audible, and taps it with the level-meter worklet. Only chrome.runtime is available here.
import workletUrl from '../worklet/level-meter.worklet.ts?worker&url';
import { LEVEL_METER_PROCESSOR } from '../shared/level';
import {
  isLevel,
  isMessage,
  LEVELS_PORT,
  settle,
  type LevelMessage,
  type Message,
  type Reply,
} from '../shared/messages';

interface Session {
  stream: MediaStream;
  context: AudioContext;
}

let session: Session | null = null;
const levelPorts = new Set<chrome.runtime.Port>();

// Chrome's legacy constraint syntax for tab capture isn't in the standard MediaTrackConstraints type.
function tabAudioConstraints(streamId: string): MediaTrackConstraints {
  return {
    mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId },
  } as unknown as MediaTrackConstraints;
}

async function start(streamId: string): Promise<void> {
  await stop();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: tabAudioConstraints(streamId),
    video: false,
  });
  // Native sample rate on purpose: the passthrough below must not be resampled to 16 kHz.
  // Downsampling for speech recognition belongs inside the pipeline (Phase 3).
  const context = new AudioContext();
  session = { stream, context };
  try {
    await context.audioWorklet.addModule(workletUrl);
    const source = context.createMediaStreamSource(stream);
    // Capturing a tab stops it playing to the user; route it back out at unity gain.
    source.connect(context.destination);

    const meter = new AudioWorkletNode(context, LEVEL_METER_PROCESSOR);
    meter.port.onmessage = (event: MessageEvent<unknown>) => {
      if (isLevel(event.data)) broadcastLevel({ type: 'level', ...event.data });
    };
    source.connect(meter);
    // The meter's output is silent; connecting it keeps the node in the rendered graph.
    meter.connect(context.destination);

    for (const track of stream.getAudioTracks()) {
      track.addEventListener('ended', onTrackEnded, { once: true });
    }
  } catch (error) {
    await stop();
    throw error;
  }
}

async function stop(): Promise<void> {
  const current = session;
  if (!current) return;
  session = null;
  for (const track of current.stream.getTracks()) {
    track.removeEventListener('ended', onTrackEnded);
    track.stop();
  }
  await current.context.close();
}

function onTrackEnded(): void {
  void notifyServiceWorker({ type: 'offscreen/track-ended' });
}

async function notifyServiceWorker(message: Message): Promise<void> {
  try {
    await chrome.runtime.sendMessage(message);
  } catch (error) {
    console.error('[tabscribe] could not notify service worker', error);
  }
}

function broadcastLevel(message: LevelMessage): void {
  for (const port of levelPorts) {
    try {
      port.postMessage(message);
    } catch {
      levelPorts.delete(port);
    }
  }
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.sender?.id !== chrome.runtime.id || port.name !== LEVELS_PORT) {
    port.disconnect();
    return;
  }
  levelPorts.add(port);
  port.onDisconnect.addListener(() => levelPorts.delete(port));
});

chrome.runtime.onMessage.addListener(
  (message: unknown, sender, sendResponse: (reply: Reply) => void) => {
    if (sender.id !== chrome.runtime.id || !isMessage(message)) return false;
    switch (message.type) {
      case 'offscreen/start':
        void settle(start(message.streamId)).then(sendResponse);
        return true;
      case 'offscreen/stop':
        void settle(stop()).then(sendResponse);
        return true;
      default:
        return false;
    }
  },
);
