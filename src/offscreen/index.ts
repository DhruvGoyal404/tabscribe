// Offscreen document (reasons USER_MEDIA and WORKERS). Turns a tab stream id into audio: plays it
// back so the tab stays audible, taps it with the level meter, and captions it (captions.ts).
// Only chrome.runtime is available here.
import workletUrl from '../worklet/capture.worklet.ts?worker&url';
import type { CaptionPlan } from '../asr/backend';
import type { Caption, CaptionMessage, CaptionStatus } from '../shared/captions';
import { LEVEL_METER_PROCESSOR } from '../shared/level';
import {
  isLevel,
  isMessage,
  SESSION_PORT,
  settle,
  type LevelMessage,
  type Message,
  type Reply,
} from '../shared/messages';
import { CaptionPipeline } from './captions';
import { passthroughGain, rampPassthrough } from './passthrough';

interface Session {
  stream: MediaStream;
  context: AudioContext;
  passthrough: GainNode;
  captions: CaptionPipeline | null;
}

let session: Session | null = null;
const sessionPorts = new Set<chrome.runtime.Port>();

/** What a popup that connects mid-session needs to catch up. */
const latest: { status: CaptionStatus | null; final: Caption | null; partial: Caption | null } = {
  status: null,
  final: null,
  partial: null,
};

// Chrome's legacy constraint syntax for tab capture isn't in the standard MediaTrackConstraints type.
function tabAudioConstraints(streamId: string): MediaTrackConstraints {
  return {
    mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId },
  } as unknown as MediaTrackConstraints;
}

async function start(
  streamId: string,
  muted: boolean,
  plan: CaptionPlan,
  allowDownload: boolean,
): Promise<void> {
  await stop();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: tabAudioConstraints(streamId),
    video: false,
  });
  // Native sample rate on purpose: the passthrough below must not be resampled to 16 kHz.
  // Downsampling for speech recognition belongs inside the pipeline (Phase 3).
  const context = new AudioContext();
  // Capturing a tab stops it playing to the user, so we route it back out ourselves. Muting the
  // tab no longer silences that playback, so the passthrough follows the tab's mute state. The
  // meter taps the source before this gain: levels (and later captions) keep flowing while muted.
  const passthrough = new GainNode(context, { gain: passthroughGain(muted) });
  session = { stream, context, passthrough, captions: null };
  try {
    await context.audioWorklet.addModule(workletUrl);
    const source = context.createMediaStreamSource(stream);
    source.connect(passthrough).connect(context.destination);

    const meter = new AudioWorkletNode(context, LEVEL_METER_PROCESSOR);
    meter.port.onmessage = (event: MessageEvent<unknown>) => {
      if (isLevel(event.data)) broadcast({ type: 'level', ...event.data });
    };
    source.connect(meter);
    // The meter's output is silent; connecting it keeps the node in the rendered graph.
    meter.connect(context.destination);

    for (const track of stream.getAudioTracks()) {
      track.addEventListener('ended', onTrackEnded, { once: true });
    }

    // Capture is up; the model may take a while (or a download), so captions follow on their own.
    session.captions = CaptionPipeline.start(context, source, plan, allowDownload, {
      status: (status) => {
        latest.status = status;
        broadcast({ type: 'captions/status', status });
      },
      caption: (caption) => {
        if (caption.kind === 'final') {
          latest.final = caption;
          latest.partial = null;
        } else latest.partial = caption;
        broadcast({ type: 'captions/caption', caption });
      },
      webgpuFailed: (message) => {
        void notifyServiceWorker({ type: 'offscreen/webgpu-failed', message });
      },
    });
  } catch (error) {
    await stop();
    throw error;
  }
}

async function stop(): Promise<void> {
  const current = session;
  if (!current) return;
  session = null;
  // Terminates the VAD and ASR workers, which unloads the model.
  current.captions?.stop();
  latest.status = null;
  latest.final = null;
  latest.partial = null;
  for (const track of current.stream.getTracks()) {
    track.removeEventListener('ended', onTrackEnded);
    track.stop();
  }
  await current.context.close();
}

function setMuted(muted: boolean): void {
  if (!session) return;
  rampPassthrough(session.passthrough.gain, muted, session.context.currentTime);
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

type SessionMessage = LevelMessage | CaptionMessage;

function send(port: chrome.runtime.Port, message: SessionMessage): void {
  try {
    port.postMessage(message);
  } catch {
    sessionPorts.delete(port);
  }
}

function broadcast(message: SessionMessage): void {
  for (const port of sessionPorts) send(port, message);
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.sender?.id !== chrome.runtime.id || port.name !== SESSION_PORT) {
    port.disconnect();
    return;
  }
  sessionPorts.add(port);
  port.onDisconnect.addListener(() => sessionPorts.delete(port));
  if (latest.status) send(port, { type: 'captions/status', status: latest.status });
  for (const caption of [latest.final, latest.partial]) {
    if (caption) send(port, { type: 'captions/caption', caption });
  }
});

chrome.runtime.onMessage.addListener(
  (message: unknown, sender, sendResponse: (reply: Reply) => void) => {
    if (sender.id !== chrome.runtime.id || !isMessage(message)) return false;
    switch (message.type) {
      case 'offscreen/start':
        void settle(
          start(message.streamId, message.muted, message.plan, message.allowDownload),
        ).then(sendResponse);
        return true;
      case 'offscreen/set-muted':
        setMuted(message.muted);
        sendResponse({ ok: true });
        return false;
      case 'offscreen/stop':
        void settle(stop()).then(sendResponse);
        return true;
      default:
        return false;
    }
  },
);
