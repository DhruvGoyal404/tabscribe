# P2 spike: tab audio capture

**Goal:** prove the capture path end to end. Popup Start → tab audio captured → level meter in the
popup → tab stays audible. No VAD, ASR or overlay yet.

**Result:** the path works, including on YouTube, embeds and a DRM site, with no audible change in
quality or latency. The manual matrix found two bugs (tab mute ignored, badge in every window). Both
are fixed below and need a re-test on Chrome stable.

## Architecture

```text
popup ──capture/start {tabId, streamId}──▶ service worker
  tabCapture.getMediaStreamId      │ tabs.get(tabId).mutedInfo
  ({ targetTabId }), since P3b     │
                                  │ offscreen.createDocument(USER_MEDIA), one shared in-flight promise
                                  ▼
                           offscreen document ◀── offscreen/start {streamId, muted}
                                  │ getUserMedia({ chromeMediaSource: 'tab' })        offscreen/set-muted
                                  │ AudioContext (native rate)                        (tabs.onUpdated)
                                  │   source ──▶ GainNode ──▶ destination   (passthrough, follows tab mute)
                                  │   source ──▶ level-meter worklet ──▶ destination (silent, pre-gain)
                                  ▼
popup ◀──── port "levels" ──── { rmsDbfs, peakDbfs, clipped } at ~20 Hz
```

- **State:** the service worker owns `idle | starting | capturing | stopping | error`
  ([capture-state.ts](../../src/shared/capture-state.ts)) and keeps it in `chrome.storage.session`
  rather than module variables, because the worker can restart at any time. The popup renders from
  storage and subscribes to `storage.session.onChanged`. Operations are serialized, so double clicks
  and overlapping triggers can't interleave.
- **Teardown** runs on user Stop, `tabs.onRemoved` for the captured tab, the track's `ended` event,
  and errors. It stops tracks, closes the AudioContext, closes the offscreen document, and resets
  state. Starting on another tab stops the first capture cleanly, then starts the new one.
- **Badge:** `ON` on the captured tab only, derived from the state on every transition
  ([tab-sync.ts](../../src/background/tab-sync.ts)).
- **Messages:** a discriminated union with a hand-written guard
  ([messages.ts](../../src/shared/messages.ts)). Unknown shapes and extra keys are rejected. Every
  `onMessage` and `onConnect` handler checks `sender.id === chrome.runtime.id`.
- **Levels** go from the offscreen document to the popup over a `runtime.connect` port, not broadcast
  `sendMessage`. Nothing is sent while the popup is closed, and the service worker isn't woken 20
  times a second. The popup reconnects whenever the state is `capturing`, so reopening it mid-capture
  resumes the meter.

### How the worklet gets into `dist/`

`src/offscreen/index.ts` imports `../worklet/level-meter.worklet.ts?worker&url`. Vite bundles that
file, with `src/shared/level.ts` inlined, into a standalone ES module (`worker.format: 'es'`) at
`dist/assets/level-meter.worklet-<hash>.js`. The import resolves to the root-relative URL
`/assets/level-meter.worklet-<hash>.js`, which `audioWorklet.addModule` loads as
`chrome-extension://<id>/assets/…`. That is `'self'` under our CSP, with no `blob:` or `data:` URL. The
probe below confirms it loads and runs: levels arrive.

The worklet has its own `tsconfig.json` (no DOM lib) and minimal AudioWorkletGlobalScope
declarations, so DOM APIs can't leak into code running on the audio thread.

## Finding: the user-gesture question

> **Changed in P3b:** the popup now requests the stream id and sends it with `capture/start`. Once
> the extension became cross-origin isolated (for multithreaded ASR), an id issued by the service
> worker, which isn't isolated, fails in the isolated offscreen document with `AbortError: Error
starting tab capture`. See [p3b-streaming.md](./p3b-streaming.md#cross-origin-isolation-multithreaded-wasm). The
> activeTab finding below still holds: the popup is the invoked context.

**`chrome.tabCapture.getMediaStreamId({ targetTabId })` called in the service worker, after a message
from the popup, works.** No popup-side fallback is needed.

Why: the requirement isn't a transient user gesture that has to survive the message hop. It's an
**invocation grant**. The [tabCapture docs](https://developer.chrome.com/docs/extensions/reference/api/tabCapture)
say `targetTabId` must be a tab "for which the extension has been granted the `activeTab`
permission", which happens when the user invokes the extension, for example by clicking the action.
Opening the popup is that invocation, and the grant lasts until the tab navigates or closes.

Verified empirically with the [probe](./p2-capture-probe.mjs) on Chromium 156:

| Case                                                                                                                            | Result                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Popup URL opened as an ordinary tab (not an invocation), Start for another tab                                                  | `Error: Extension has not been invoked for the current page (see activeTab permission). Chrome pages cannot be captured.` State goes to `error`, and teardown leaves nothing behind. |
| Popup opened by running the action on the tab (CDP `Extensions.triggerAction`, the same path as a toolbar click), Start clicked | `capturing`, badge `ON`, one offscreen document, levels flowing                                                                                                                      |

Caveat: after the tab navigates, Chrome drops the activeTab grant. Starting a _new_ capture then needs
the popup opened again on that page, which happens naturally because Start lives in the popup.

## Fixes from the manual matrix

### Tab mute was ignored (matrix #7)

Capturing a tab moves its playback into the offscreen document, so _Mute site_ marked the tab as
muted but you still heard the audio. Captions while muted is a real use case, so capture keeps
running and only what you hear follows the mute:

- The passthrough goes through a `GainNode`. The level tap connects before it, so the meter (and
  later the ASR pipeline) still gets audio while the tab is muted.
- The service worker reads the initial `mutedInfo` with `tabs.get(tabId)` at start and sends it with
  `offscreen/start`. Later changes come from `tabs.onUpdated` `changeInfo.mutedInfo` for the captured
  tab only, and go to the offscreen document as `offscreen/set-muted`.
- Gain changes ramp with `setTargetAtTime` (time constant 5 ms, within 2% after ~20 ms), so there's
  no click.
- **No new permission.** The [tabs docs](https://developer.chrome.com/docs/extensions/reference/api/tabs)
  put only `url`, `pendingUrl`, `title` and `favIconUrl` behind `"tabs"`. The probe confirms it with
  our manifest: `tabs.get` returned `{"muted":true,"reason":"extension",…}`, and `onUpdated` delivered
  `mutedInfo` changes `[true, false]`.

### Badge showed in every window (matrix #11)

The badge was global. It is now set with `tabId` on the captured tab only, from a diff of the capture
state on every transition: set on reaching `capturing`, cleared on stop, failure, tab close, and
when **Capture this tab** switches to another tab.

The [action docs](https://developer.chrome.com/docs/extensions/reference/api/action) only say a
per-tab value "automatically resets when the tab is closed". **Empirically, Chrome also clears it
when the tab navigates**, both same-origin and cross-origin (probe, Chromium 156), while capture
survives navigation. So the service worker re-applies the badge on the captured tab's `onUpdated`
`status` changes (`loading` and `complete`). The probe confirms `ON` on the captured tab after both
navigations, nothing on the old tab after a switch, and no badge anywhere after Stop or tab close.

## Manual matrix

**How to run:** `npm run build`, load `dist/` unpacked (see README), run `npm run serve:test-pages`
and open <http://127.0.0.1:8137/>. "Probe" means verified by
`node docs/spikes/p2-capture-probe.mjs <chromium.exe>` in headless Chromium 156, which can measure
levels but can't listen. "Manual" means tested by hand on Chrome stable (Windows) unless noted.

| #   | Case                                             | Status                | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --- | ------------------------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Plain `<audio>` on the local test page           | ✅ Probe + manual     | The meter tracked the tone exactly: −23.0 / −13.0 / −3.0 / −43.0 dBFS, then silence (the expected RMS of the −20/−10/0/−40 dBFS-peak sine steps). Manual: −13.0 dBFS at the −10 dBFS-peak step.                                                                                                                                                                                                                                                                                                               |
| 2   | Popup closed then reopened mid-capture           | ✅ Probe              | The reopened popup shows "Capturing this tab" and the meter resumes within about 1 s.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 3   | Switching to another tab while capturing         | ✅ Probe              | Capture keeps running on the original tab. On the second tab the popup shows "Capturing another tab" with **Capture this tab**, which stops the first capture and starts the second: still exactly one offscreen document, and the badge moves to the new tab.                                                                                                                                                                                                                                                |
| 4   | In-tab navigation                                | ✅ Probe              | Capture survives same-origin and cross-origin navigation, and the meter reads the new page's audio. The badge is re-applied after each navigation (see the fixes).                                                                                                                                                                                                                                                                                                                                            |
| 5   | Closing the captured tab                         | ✅ Probe              | State `idle`, 0 offscreen documents, badge cleared.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 6   | User Stop                                        | ✅ Probe              | State `idle`, 0 offscreen documents, no badge on any tab.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 7   | Tab muted by the user (_Mute site_)              | 🔧 Fixed, **re-test** | **Bug (manual):** the tab showed as muted but stayed audible through our passthrough, and the meter kept moving. **Fixed:** the passthrough gain now follows the tab's mute state, and the meter keeps running because it is tapped before the gain. **Re-test:** mute and unmute while the tone plays. Audio should cut out and come back with no click, and the meter should keep moving throughout. Muting _inside_ the YouTube player silences the meter, which is expected: the source itself is silent. |
| 8   | Tab stays audible; added latency or quality loss | ✅ Manual             | No audible difference in quality and no lip-sync drift with capture on vs off.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 9   | YouTube video                                    | ✅ Manual             | The meter moves and the audio stays audible.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 10  | YouTube iframe embed on the test page            | ✅ Manual             | The embed's audio is in the tab mix, and the meter moves.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 11  | Two windows                                      | 🔧 Fixed, **re-test** | Manual: window B's popup correctly showed "Capturing another tab" with **Capture this tab**, and Stop stopped the single capture (both popups went to Idle). **UX bug:** the `ON` badge showed on the action icon in both windows, so you couldn't tell which tab was captured. **Fixed:** the badge is per tab. **Re-test:** capture in window A, and window B's icon has no badge. **Capture this tab** in B moves the badge to B's tab.                                                                    |
| 12  | DRM site                                         | ✅ Manual (Brave)     | Prime Video in Brave (Chromium): audio **is** captured, the meter moves, and the tab stays audible. DRM blocks the video, not the audio.                                                                                                                                                                                                                                                                                                                                                                      |
| 13  | CPU usage                                        | ✅ Manual             | Chrome Task Manager while capturing YouTube: "Extension: Tabscribe" at **2.3–4.4% CPU, ~37 MB**. See the baseline below.                                                                                                                                                                                                                                                                                                                                                                                      |

### CPU baseline (for comparison after P3)

P2 is capture, passthrough and a level meter only: **2.3–4.4% CPU and ~37 MB** for the "Extension:
Tabscribe" row in Chrome's Task Manager while capturing YouTube on the developer's Windows machine. P3
adds resampling, VAD and Whisper inference in the same offscreen document. Measure the same row with
the same machine and video, so the cost of ASR shows against this number.

## Not covered (by design)

- No VAD, ASR, overlay or 16 kHz resampling. Resampling belongs inside the P3 pipeline, not on the
  AudioContext, so the passthrough stays full quality.
- e2e only checks that the popup renders the idle state. Real `tabCapture` isn't automated in CI. The
  probe is a local tool and needs `--enable-unsafe-extension-debugging`. It can't listen, so for the mute
  fix it only confirms that the service worker receives `mutedInfo`. Whether the audio actually goes
  silent needs the manual re-test.
