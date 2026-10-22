#!/usr/bin/env node
// Serves test-pages/ on http://127.0.0.1:<PORT> (default 8137) for the manual capture matrix.
// /tone.wav is generated on the fly: a 440 Hz tone stepping through levels so the meter moves.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../test-pages');
const PORT = Number(process.env.PORT ?? 8137);
const TYPES = { '.html': 'text/html; charset=utf-8', '.css': 'text/css', '.js': 'text/javascript' };

// Amplitudes per 2-second step: -20 dBFS, -10 dBFS, full scale, -40 dBFS, silence.
const STEPS = [0.1, 0.316, 1, 0.01, 0];

function toneWav(sampleRate = 48000, secondsPerStep = 2) {
  const frames = sampleRate * secondsPerStep * STEPS.length;
  const buffer = Buffer.alloc(44 + frames * 2);
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + frames * 2, 4);
  buffer.write('WAVEfmt ', 8);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(frames * 2, 40);
  for (let i = 0; i < frames; i++) {
    const amplitude = STEPS[Math.floor(i / (sampleRate * secondsPerStep))] ?? 0;
    const sample = amplitude * Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    buffer.writeInt16LE(Math.round(Math.max(-1, Math.min(1, sample)) * 0x7fff), 44 + i * 2);
  }
  return buffer;
}

const TONE = toneWav();

const server = createServer(async (req, res) => {
  const { pathname } = new URL(req.url ?? '/', 'http://localhost');
  if (pathname === '/tone.wav') {
    res.writeHead(200, { 'content-type': 'audio/wav', 'content-length': TONE.length });
    res.end(TONE);
    return;
  }
  const file = path.resolve(ROOT, `.${pathname === '/' ? '/index.html' : pathname}`);
  const type = TYPES[path.extname(file)];
  if (!file.startsWith(ROOT + path.sep) || !type) {
    res.writeHead(404).end('Not found');
    return;
  }
  try {
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': type }).end(body);
  } catch {
    res.writeHead(404).end('Not found');
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Test pages: http://127.0.0.1:${String(PORT)}/`);
});
