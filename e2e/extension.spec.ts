import { expect, test } from './fixtures';

test('extension service worker starts', ({ extensionId }) => {
  expect(extensionId).toMatch(/^[a-p]{32}$/);
});

// Opening the popup page directly isn't an action invocation, so this only checks rendering;
// real tab capture needs a user click and is covered by the manual matrix (docs/spikes).
test('popup renders the idle state', async ({ context, extensionId }) => {
  const page = await context.newPage();
  await page.goto(`chrome-extension://${extensionId}/src/popup/index.html`);

  await expect(page.getByRole('heading', { name: 'Tabscribe' })).toBeVisible();
  await expect(page.getByRole('status')).toHaveText('Idle');
  await expect(page.getByRole('button', { name: 'Start' })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Stop' })).toBeDisabled();
  await expect(page.getByRole('meter', { name: 'Tab audio level' })).toHaveAttribute(
    'value',
    '-60',
  );
  await expect(page.getByRole('alert')).toBeHidden();
  await expect(page.getByText(/^v\d+\.\d+\.\d+$/)).toBeVisible();
});
