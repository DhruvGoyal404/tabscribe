import { expect, test } from './overlay-fixtures';

// Page-side checks are strings, as e2e/ has no DOM types.
const HOSTS = `document.querySelectorAll('tabscribe-overlay').length`;

const listening = {
  type: 'captions/status',
  status: {
    state: 'listening',
    plan: {
      device: 'wasm',
      dtype: { encoder: 'q8', decoder: 'q8' },
      threads: 4,
      reason: 'e2e',
    },
    vad: 'silero',
  },
};
const caption = (kind: 'partial' | 'final', segmentId: number, text: string) => ({
  type: 'captions/caption',
  caption: { kind, segmentId, text },
});

test('overlay shows captions in a closed shadow root, and goes when the session ends', async ({
  page,
  session,
  inject,
  overlay,
}) => {
  const violations: string[] = [];
  page.on('console', (message) => {
    if (/Content Security Policy/i.test(message.text())) violations.push(message.text());
  });
  await inject();
  await session.connected();
  // Nothing to show yet.
  await expect.poll(overlay).toMatchObject({ shadowRootType: 'closed', hidden: true });

  await session.send(listening);
  await session.send(caption('partial', 0, 'He has'));
  await expect.poll(overlay).toMatchObject({ hidden: false, final: '', partial: 'He has' });
  await session.send(caption('final', 0, 'He has grave doubts.'));
  await session.send(caption('partial', 1, '<b>whether</b>'));
  // Text only: markup in a caption stays text.
  await expect
    .poll(overlay)
    .toMatchObject({ final: 'He has grave doubts.', partial: '<b>whether</b>' });

  // The page can't reach into the overlay, and its CSS can't hide the host.
  expect(await page.evaluate<boolean>(`document.querySelector('tabscribe-overlay').shadowRoot === null`)).toBe(true);
  expect(await page.evaluate<string>(`getComputedStyle(document.querySelector('tabscribe-overlay')).display`)).not.toBe('none');
  // The page's strict CSP didn't block the overlay's styles.
  expect(violations).toEqual([]);

  await session.close();
  await expect.poll(() => page.evaluate<number>(HOSTS)).toBe(0);
}); // prettier-ignore

test('injecting again replaces the overlay instead of adding one', async ({
  page,
  session,
  inject,
}) => {
  await inject();
  await session.connected();
  await inject();
  await session.connected(2);
  await expect.poll(() => page.evaluate<number>(HOSTS)).toBe(1);
});

test('a status other than listening clears the overlay', async ({ session, inject, overlay }) => {
  await inject();
  await session.connected();
  await session.send(listening);
  await session.send(caption('final', 0, 'Hello there.'));
  await expect.poll(overlay).toMatchObject({ hidden: false, final: 'Hello there.' });
  await session.send({
    type: 'captions/status',
    status: { state: 'error', plan: listening.status.plan, message: 'out of memory' },
  });
  await expect.poll(overlay).toMatchObject({ hidden: true, final: '' });
});
