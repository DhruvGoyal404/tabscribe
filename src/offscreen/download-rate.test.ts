import { describe, expect, it } from 'vitest';
import { DownloadRate } from './download-rate';

describe('DownloadRate', () => {
  it('has no rate until a second has passed', () => {
    const rate = new DownloadRate();
    expect(rate.update(0, 0)).toBeNull();
    expect(rate.update(500, 0.3e6)).toBeNull();
    expect(rate.update(2_000, 1.2e6)).toBe(0.6e6);
  });

  it('follows the last ten seconds', () => {
    const rate = new DownloadRate();
    rate.update(0, 0);
    rate.update(10_000, 10e6); // 1 MB/s
    rate.update(20_000, 15e6); // then 0.5 MB/s
    expect(rate.update(30_000, 20e6)).toBe(0.5e6);
  });
});
