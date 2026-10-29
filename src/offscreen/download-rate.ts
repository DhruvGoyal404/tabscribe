/** How fast a download is going, over its last few seconds (a whole-download average would lag). */
export class DownloadRate {
  private readonly samples: { ms: number; bytes: number }[] = [];

  constructor(
    private readonly windowMs = 10_000,
    /** Shorter spans give no rate: the first progress events come in bursts. */
    private readonly minSpanMs = 1_000,
  ) {}

  /** Records `bytes` loaded so far at `ms`; returns bytes per second, or null while unknown. */
  update(ms: number, bytes: number): number | null {
    this.samples.push({ ms, bytes });
    while (this.samples.length > 2 && ms - (this.samples[1]?.ms ?? ms) >= this.windowMs) {
      this.samples.shift();
    }
    const first = this.samples[0];
    if (!first || ms - first.ms < this.minSpanMs) return null;
    return Math.max(0, ((bytes - first.bytes) * 1000) / (ms - first.ms));
  }
}
