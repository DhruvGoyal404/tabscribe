# P2 spike: tab audio capture

**Goal:** prove the capture path end to end. Popup Start → tab audio captured → level meter in the
popup → tab stays audible. No VAD, ASR or overlay yet.

**Result:** the path works. The open questions are what the tab sounds like while captured (needs a
human ear) and behaviour on DRM sites.

## Architecture

```text
popup ──capture/start {tabId}──▶ service worker
                                  │ tabCapture.getMediaStreamId({ targetTabId })
                                  │ offscreen.createDocument(USER_MEDIA), one shared in-flight promise
                                  ▼
                           offscreen document ◀── offscreen/start {streamId}
                                  │ getUserMedia({ chromeMediaSource: 'tab' })
                                  │ AudioContext (native rate)
                                  │   source ──▶ destination          (passthrough, unity gain)
                                  │   source ──▶ level-meter worklet ──▶ destination (silent)
                                  ▼
popup ◀──── port "levels" ──── { rmsDbfs, peakDbfs, clipped } at ~20 Hz
```

- **State:** the service worker owns `idle | starting | capturing | stopping | error`
  ([capture-state.ts](../../src/shared/capture-state.ts)) and keeps it in `chrome.storage.session`
  rather than module variables, because the worker can restart at any time. The popup renders from
  storage and subscribes to `storage.session.onChanged`. Operations are serialized, so double clicks
  and overlapping triggers can't interleave.
- **Teardown** runs on user Stop, `tabs.onRemoved` for the captured tab, the track's `ended` event,
  and errors. It stops tracks, closes the AudioContext, closes the offscreen document, clears the
  badge, and resets state. Starting on another tab stops the first capture cleanly, then starts the
  new one.
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

## Manual matrix

**How to run:** `npm run build`, load `dist/` unpacked (see README), run `npm run serve:test-pages`
and open <http://127.0.0.1:8137/>. "Probe" means verified by
`node docs/spikes/p2-capture-probe.mjs <chromium.exe>` in headless Chromium 156, which can measure
levels but can't listen.

| #   | Case                                             | Status             | Result / steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --- | ------------------------------------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Plain `<audio>` on the local test page           | ✅ Probe           | The meter tracked the tone exactly: −23.0 / −13.0 / −3.0 / −43.0 dBFS, then silence (expected RMS of the −20/−10/0/−40 dBFS-peak sine steps).                                                                                                                                                                                                                                                                                                                                                     |
| 2   | Popup closed then reopened mid-capture           | ✅ Probe           | Reopened popup shows "Capturing this tab" and the meter resumes within about 1 s.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 3   | Switching to another tab while capturing         | ✅ Probe (state)   | Capture keeps running on the original tab. On the second tab the popup shows "Capturing another tab" with **Capture this tab**. Clicking it stops the first capture and starts the second: still exactly one offscreen document.                                                                                                                                                                                                                                                                  |
| 4   | In-tab navigation                                | ✅ Probe           | Capture survives both same-origin and cross-origin navigation in the captured tab, and the meter keeps reading the new page's audio.                                                                                                                                                                                                                                                                                                                                                              |
| 5   | Closing the captured tab                         | ✅ Probe           | State `idle`, 0 offscreen documents, badge cleared.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 6   | User Stop                                        | ✅ Probe           | State `idle`, 0 offscreen documents, badge cleared.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 7   | Tab muted by the user                            | ⚠️ Probe + **you** | **Surprise:** muting the tab does _not_ affect capture. Levels kept following the full tone cycle while muted, and `tab.audible` stayed true. Playback comes from the offscreen document's own AudioContext, so muting the tab probably doesn't silence what you hear. **Steps:** start capture on the test page, right-click the tab → _Mute site_, and listen. If the audio is still audible, the fix (P3/P4) is to watch `tabs.onUpdated` `mutedInfo` and drive a passthrough `GainNode` to 0. |
| 8   | Tab stays audible; added latency or quality loss | 🔲 **You**         | Headless can't listen. **Steps:** play the test tone, note how it sounds, start capture, and listen for a gap, echo, lower quality or added delay. Then do the same with music on YouTube. Expected: unchanged quality (the context runs at the device's native rate), with possibly a few ms of output-buffer latency.                                                                                                                                                                           |
| 9   | YouTube video                                    | 🔲 **You**         | Open any YouTube video, play it, open the popup on that tab → Start. Check: meter moves, audio stays audible, badge `ON`. Then try fullscreen and an ad break.                                                                                                                                                                                                                                                                                                                                    |
| 10  | YouTube iframe embed on the test page            | 🔲 **You**         | Play the embed on the test page and start capture. Tab capture records the whole tab's mix, so the iframe should be included. Check that the meter moves.                                                                                                                                                                                                                                                                                                                                         |
| 11  | Two windows                                      | 🔲 **You**         | Capture a tab in window A, focus window B, open the popup there. Expect "Capturing another tab", and **Capture this tab** switches over. Closing window A should tear down (same as #5).                                                                                                                                                                                                                                                                                                          |
| 12  | DRM site (Netflix, Prime Video or Spotify web)   | 🔲 **You**         | Start capture on a playing title. Record: does the meter move, and do you still hear audio? Some DRM pipelines deliver silent or blocked audio to captures. If it's silent, the popup should still show `capturing`, and that's a product note, not a bug.                                                                                                                                                                                                                                        |
| 13  | CPU usage                                        | 🔲 **You**         | Chrome menu → _More tools_ → _Task manager_. Note CPU% for the "Tabscribe" offscreen and extension rows, and the "Audio Service" utility process, while capturing and while idle. My headless attempt via CDP `SystemInfo.getProcessInfo` gave identical numbers for both, which isn't credible, so I dropped it.                                                                                                                                                                                 |

## Not covered (by design)

- No VAD, ASR, overlay or 16 kHz resampling. Resampling belongs inside the P3 pipeline, not on the
  AudioContext, so the passthrough stays full quality.
- e2e only checks that the popup renders the idle state. Real `tabCapture` isn't automated in CI. The
  probe is a local tool and needs `--enable-unsafe-extension-debugging`.
