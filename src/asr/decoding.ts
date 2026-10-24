// Hooks into Transformers.js's generate() loop for one guarded Whisper decode.
//
// generate() runs each step's logits through a list of processors (most edit them in place), lets
// the sampler pick a token, then asks the stopping criteria whether to go on.
import { LogitsProcessor, StoppingCriteria, type Tensor } from '@huggingface/transformers';

/** Batch row 0 of a [batch, vocab] logits tensor, as a view that in-place edits go through. */
function row(logits: Tensor): Float32Array {
  const vocab = logits.dims.at(-1) ?? 0;
  return (logits.data as Float32Array).subarray(0, vocab);
}

/**
 * Bans a second occurrence of any `size`-token sequence among the generated tokens. Transformers.js's
 * own no_repeat_ngram_size also counts the prompt, which would stop Whisper from transcribing the
 * words that a forced cut's overlap repeats from the previous segment's text.
 */
export class GeneratedNoRepeatNGram extends LogitsProcessor {
  constructor(
    private readonly size: number,
    private readonly promptLength: number,
  ) {
    super();
  }

  override _call(inputIds: bigint[][], logits: Tensor): Tensor {
    const generated = inputIds[0]?.slice(this.promptLength) ?? [];
    const context = this.size - 1;
    if (generated.length < context) return logits;
    const key = generated.slice(generated.length - context);
    const scores = row(logits);
    for (let start = 0; start + context < generated.length; start++) {
      if (key.every((token, i) => generated[start + i] === token)) {
        scores[Number(generated[start + context])] = -Infinity;
      }
    }
    return logits;
  }
}

/** log_softmax(scores)[token], with the scores first multiplied by `scale`. */
export function logProbability(scores: Float32Array, token: number, scale = 1): number {
  let max = -Infinity;
  for (const score of scores) max = Math.max(max, score * scale);
  let sum = 0;
  for (const score of scores) sum += Math.exp(score * scale - max);
  return (scores[token] ?? -Infinity) * scale - max - Math.log(sum);
}

/**
 * Sums the log-probability of each generated token, as openai/whisper does for avg_logprob: from the
 * logits after suppression, before temperature. Must be the last processor added. Whisper's
 * generate() appends its own processor after it, so each step's logits are kept by reference and
 * read once the next token is known, after every processor has edited them.
 */
export class TokenLogprobs extends LogitsProcessor {
  private pending: Float32Array | null = null;
  private sum = 0;
  private count = 0;

  /** `temperature`: what the sampler divided the logits by (0 for greedy). */
  constructor(private readonly temperature: number) {
    super();
  }

  override _call(inputIds: bigint[][], logits: Tensor): Tensor {
    this.settle(inputIds[0]?.at(-1));
    this.pending = row(logits);
    return logits;
  }

  /** The mean over generated tokens; pass the last token of the finished sequence. */
  average(lastToken: bigint | undefined): number {
    this.settle(lastToken);
    return this.count > 0 ? this.sum / this.count : -Infinity;
  }

  private settle(token: bigint | undefined): void {
    if (!this.pending || token === undefined) return;
    this.sum += logProbability(this.pending, Number(token), this.temperature || 1);
    this.count += 1;
    this.pending = null;
  }
}

/** Stops generate() at the next step once `requested` is set (a final is waiting on a partial). */
export class Interrupt extends StoppingCriteria {
  requested = false;

  // Typed as number[][] by Transformers.js; the ids are bigints at runtime, and unused here.
  override _call(inputIds: unknown[][]): boolean[] {
    return inputIds.map(() => this.requested);
  }
}
