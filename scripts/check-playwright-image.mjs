#!/usr/bin/env node
// Fails if the Playwright container image in CI doesn't match the @playwright/test version.
// The image ships browsers built for one Playwright release; a mismatch fails e2e with a confusing
// "Executable doesn't exist" error. Also requires the image to be pinned by digest.
import { readFile } from 'node:fs/promises';

const read = async (path) => readFile(new URL(`../${path}`, import.meta.url), 'utf8');

const pkg = JSON.parse(await read('package.json'));
const lock = JSON.parse(await read('package-lock.json'));
const workflow = await read('.github/workflows/ci.yml');

const declared = pkg.devDependencies?.['@playwright/test'];
const installed = lock.packages?.['node_modules/@playwright/test']?.version;
const images = [
  ...workflow.matchAll(
    /mcr\.microsoft\.com\/playwright:v([^\s@-]+)(-[a-z]+)?(@sha256:[0-9a-f]{64})?/g,
  ),
];

const problems = [];
if (!declared) problems.push('@playwright/test is not in devDependencies.');
if (declared !== installed) {
  problems.push(
    `package.json pins @playwright/test ${declared}, package-lock.json has ${installed}.`,
  );
}
if (images.length === 0) {
  problems.push('No mcr.microsoft.com/playwright image found in .github/workflows/ci.yml.');
}
for (const [image, version, , digest] of images) {
  if (version !== declared) {
    problems.push(
      `ci.yml uses ${image.split('@')[0]} but @playwright/test is ${declared}. ` +
        `Update the image tag to v${declared}-<distro> and its digest together with the package.`,
    );
  }
  if (!digest) problems.push(`ci.yml image ${image} is not pinned by @sha256 digest.`);
}

if (problems.length > 0) {
  for (const problem of problems) console.error(`✗ ${problem}`);
  process.exit(1);
}
console.log(
  `✓ Playwright image v${declared} matches @playwright/test ${declared} (digest-pinned).`,
);
