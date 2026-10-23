# Silero VAD model

`silero_vad_16k_op15.onnx` is bundled with the extension and loaded by the VAD worker from the
extension's own origin. Nothing is downloaded at runtime.

| What    | Value                                                                                 |
| ------- | ------------------------------------------------------------------------------------- |
| Source  | [snakers4/silero-vad](https://github.com/snakers4/silero-vad), `src/silero_vad/data/` |
| Version | tag `v6.2.3`, commit `5cd7945676eb32225748052e2e6a0580e4686a08` (2026-09-23)          |
| Licence | MIT, see [`LICENSE`](LICENSE) (copied unchanged from the same commit)                 |
| Size    | 1,289,603 bytes                                                                       |
| SHA-256 | `7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49`                    |
| Variant | 16 kHz only, ONNX opset 15                                                            |

Interface (checked with ONNX Runtime): inputs `input` float32 `[1, 576]`, `state` float32
`[2, 1, 128]`, `sr` int64 scalar (16000); outputs `output` float32 `[1, 1]` (speech probability)
and `stateN` (the next state). Each call takes 512 new samples (32 ms) preceded by the previous
call's last 64 samples, as the official `OnnxWrapper` does.

To update it, download the same file from a newer tag that passes the 14-day rule, update the
table above, and re-run the bench's VAD panel.
