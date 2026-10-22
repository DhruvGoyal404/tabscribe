// Runs on the audio rendering thread. Bundled as a standalone module (see the `?worker&url`
// import in the offscreen document) and loaded with audioWorklet.addModule.
import { blockSizeFor, downmix, LEVEL_METER_PROCESSOR, LevelMeter } from '../shared/level';

class LevelMeterProcessor extends AudioWorkletProcessor {
  private readonly meter = new LevelMeter(blockSizeFor(sampleRate));
  private mono = new Float32Array(128);

  override process(inputs: Float32Array[][]): boolean {
    const channels = inputs[0];
    const frames = channels?.[0]?.length ?? 0;
    if (channels && frames > 0) {
      if (this.mono.length !== frames) this.mono = new Float32Array(frames);
      // One level per ~50 ms block, which throttles posts to ~20 Hz.
      const level = this.meter.push(downmix(channels, this.mono));
      if (level) this.port.postMessage(level);
    }
    // Outputs stay zero-filled: this node is a silent tap.
    return true;
  }
}

registerProcessor(LEVEL_METER_PROCESSOR, LevelMeterProcessor);
