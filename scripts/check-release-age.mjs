#!/usr/bin/env node
// Fails if any package in package-lock.json was published fewer than MIN_DAYS days ago.
// Covers transitive dependencies, which npm's min-release-age only enforces at resolve time.
// Fails closed: a registry error or a missing publish time counts as a violation.
import { readFile } from 'node:fs/promises';

const MIN_DAYS = Number(process.env.MIN_RELEASE_AGE_DAYS ?? 14);
const REGISTRY = 'https://registry.npmjs.org/';
const CONCURRENCY = 16;

const lock = JSON.parse(await readFile(new URL('../package-lock.json', import.meta.url), 'utf8'));
const cutoff = Date.now() - MIN_DAYS * 24 * 60 * 60 * 1000;

/** @type {Map<string, Set<string>>} */
const wanted = new Map();
for (const [path, entry] of Object.entries(lock.packages ?? {})) {
  if (path === '' || entry.link || !entry.version) continue;
  if (!entry.resolved?.startsWith(REGISTRY)) {
    console.error(
      `✗ ${path}: not resolved from ${REGISTRY} (${entry.resolved ?? 'no resolved URL'})`,
    );
    process.exitCode = 1;
    continue;
  }
  const name = entry.name ?? path.slice(path.lastIndexOf('node_modules/') + 'node_modules/'.length);
  if (!wanted.has(name)) wanted.set(name, new Set());
  wanted.get(name).add(entry.version);
}

async function fetchTimes(name) {
  for (let attempt = 1; ; attempt++) {
    try {
      const res = await fetch(REGISTRY + name.replace('/', '%2f'));
      if (!res.ok) throw new Error(`HTTP ${String(res.status)}`);
      return (await res.json()).time ?? {};
    } catch (error) {
      if (attempt >= 3) throw error;
      await new Promise((resolve) => setTimeout(resolve, 500 * attempt));
    }
  }
}

const violations = [];
const names = [...wanted.keys()];
let next = 0;
await Promise.all(
  Array.from({ length: CONCURRENCY }, async () => {
    while (next < names.length) {
      const name = names[next++];
      let times;
      try {
        times = await fetchTimes(name);
      } catch (error) {
        violations.push(`${name}: registry lookup failed (${error.message})`);
        continue;
      }
      for (const version of wanted.get(name)) {
        const published = Date.parse(times[version]);
        if (Number.isNaN(published)) {
          violations.push(`${name}@${version}: no publish time in registry`);
        } else if (published > cutoff) {
          const days = ((Date.now() - published) / 86_400_000).toFixed(1);
          violations.push(`${name}@${version}: published ${times[version]} (${days} days ago)`);
        }
      }
    }
  }),
);

const total = [...wanted.values()].reduce((sum, versions) => sum + versions.size, 0);
if (violations.length > 0) {
  for (const v of violations.sort()) console.error(`✗ ${v}`);
  console.error(
    `\n${String(violations.length)} of ${String(total)} packages are younger than ${String(MIN_DAYS)} days or could not be checked.`,
  );
  process.exitCode = 1;
} else if (process.exitCode !== 1) {
  console.log(`✓ All ${String(total)} locked packages are at least ${String(MIN_DAYS)} days old.`);
}
