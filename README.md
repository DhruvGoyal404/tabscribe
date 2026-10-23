# Tabscribe

Private, on-device captions for any browser tab.

Tabscribe is a Chrome extension (Manifest V3) that captions the audio playing in a tab. Speech
recognition runs locally with [Whisper](https://github.com/openai/whisper) through
[Transformers.js](https://github.com/huggingface/transformers.js): `whisper-base` on WebGPU when
your device has a GPU adapter, and on WebAssembly otherwise.

**English first.** Other languages, and translating captions into English, will be turned on one
language at a time, and only once they pass our on-device benchmark. Hindi doesn't pass yet (see
[the benchmark](docs/spikes/p3-asr-bench.md)).

> **Status:** early development. The extension loads, but captioning isn't implemented yet.

## Privacy promise

- Audio is processed in memory on your device and is never recorded, stored or uploaded.
- No accounts, analytics or telemetry.
- The only network traffic is a one-time download of the speech model from the Hugging Face Hub,
  pinned to a fixed revision and cached locally.
- A future, **opt-in** cloud mode will use your own Hugging Face token and send audio only to
  Hugging Face, only while you have it switched on.

## Roadmap

- **P3b:** live English captions (streaming, with repetition guards), shown in the popup.
- **P3c:** Hindi spike: re-test with the guards, look for openly licensed Hindi fine-tuned
  Whisper models, or offer Hindi through the opt-in cloud mode.
- **P4:** caption overlay on the page.
- **P6:** opt-in bring-your-own-key cloud mode.

## Development

Requirements: Node.js 22.13+ and npm 11.21+ (`npm install -g npm@11.21.0`).

```sh
npm ci                     # install exact locked versions
npm run build              # build the extension into dist/
npm run lint               # ESLint (type-aware)
npm run format:check       # Prettier
npm run typecheck          # TypeScript (strict)
npm test                   # unit tests (Vitest)
npm run check:release-age  # every locked package must be >= 14 days old
npm run test:e2e           # Playwright; run `npx playwright install chromium` once first
```

### Load the extension in Chrome

1. `npm run build`
2. Open `chrome://extensions` and turn on **Developer mode**.
3. Click **Load unpacked** and select the `dist/` folder.

### Dependency policy

Versions are pinned exactly (`save-exact`), and npm only resolves versions published at least
14 days ago (`min-release-age=14`). CI re-checks every entry in the lockfile.

## License

[MIT](LICENSE)
