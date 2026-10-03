# Can a small draft model make a large one generate faster without changing a single output token?

**Status: Phases 0–4 of 6 complete (KV cache, accept-reject, draft-verify loop, latency benchmark).**

## Why tokens/sec is the wrong headline

### Where the stopwatch stops decides the TTFT

This loop emits the first token from the target's prefill, before the draft runs. Speculative decoding then costs nothing in TTFT, and the draft's one-time prefill is spread across every token of TPOT. Stopping the stopwatch at the end of the first speculative round instead moves the draft's prefill, its k steps and the first verify into TTFT. It also biases TPOT low: the round's time leaves the numerator, but its tokens stay in the denominator. `transformers`' assisted path can only be measured the second way, because it emits nothing before its first round. Both definitions are computed from the same Phase 4 run and reported side by side, with the draft prefill and end-to-end latency as separate columns. Definitions: `bench/latency.py`.

On the 135M draft, the first-token definition puts speculative TTFT within 3% of the cached baseline (706 against 689 ms on long code prompts). The first-round definition puts it 41% above (975 ms), and lowers its TPOT from 53.2 to 43.2 ms. On short prompts, where prefill is cheap, the first-round TTFT is three times the first-token one (255 against 85 ms). The assisted path's TTFT (838 ms on long code) is a first-round number and compares with 975, not with 706.

## Speed

SmolLM2-1.7B-Instruct target, SmolLM2-135M-Instruct draft, both bf16 with fp32 logits, on an Apple M3 with 8 GB (MPS). Batch size 1, k = 4, 32 new tokens, greedy. Each cell has 3 warmup rounds and 100 measured rounds, with the method order shuffled every round. Long prompts are 300–600 tokens; short ones are 9–167. Speedup is the cached baseline's end-to-end p50 divided by the method's; α is the fraction of drafted tokens accepted.

| Method | long / code | long / prose | long / repetitive | short / code | short / prose | short / repetitive |
|:--|--:|--:|--:|--:|--:|--:|
| `cached` E2E p50 (ms) | 2,726 | 2,705 | 2,847 | 1,808 | 1,796 | 1,940 |
| `no_cache` | 0.13× | 0.13× | 0.12× | 0.59× | 0.64× | 0.32× |
| `spec_ngram` | 1.32× (α 0.19) | 1.06× (α 0.05) | 1.97× (α 0.63) | 1.07× (α 0.06) | 0.98× (α 0.04) | 1.78× (α 0.32) |
| `spec_draft` | 1.17× (α 0.56) | 0.95× (α 0.32) | 1.44× (α 0.99) | 1.28× (α 0.64) | 0.98× (α 0.37) | 1.30× (α 0.69) |
| `hf_assisted` | 1.36× | 1.12× | 1.65× | 1.50× | 1.08× | 1.54× |
| `spec_self` | 0.72× | 0.73× | 0.71× | 0.84× | 0.84× | 0.80× |

- **The speedup follows the text, not the method.** On prose the draft model's acceptance falls to 0.32–0.37, and it ends up 2–5% slower than plain cached decoding on both prompt lengths. On repetitive text it reaches 1.30–1.44×, and the prompt-lookup draft, which needs no second model, reaches 1.78–1.97×. Acceptance is ordered repetitive > code > prose under both conditions.
- **The cache is the larger win, but only for long prompts.** Without it, 300–600-token prompts are 7.7–8.3× slower end to end; on 9–36-token prompts the gap is 1.6–1.7×.
- **`transformers`' assisted path beats this loop on every cell**: 1.36× against 1.17× on long code, and 1.12× against 0.95× on long prose. Both run the same models and emit the same tokens, so the difference is overhead in this implementation. The likely sources are the gather copy of every cached position per layer per step (`cache/kv.py`) and the device-to-host copy of each draft distribution. Neither has been profiled yet.
- **`spec_self` is slower everywhere (0.71–0.84×), as it must be.** It drafts with the target itself, so acceptance is 1.00 and every round pays for the model twice. A harness that showed it faster would be measuring something other than the work done.

### Sanity checks

`tests/test_bench_sanity.py` runs on this file: 252 of its checks pass and 4 fail. The failures are reported here, and their thresholds were not loosened.

- **The cache is not the first-order win on short prompts** (3 failures). The check asks for a no-cache TPOT at least 5× the cached one. On 9–36-token prompts the ratio is 1.6–1.8×, and on 37–167-token prompts it is 3.4×. There is little prefix to recompute, so caching has little to save. Long prompts pass at 10.0–10.4×. The short condition was kept to show this collapse; the check was written for long contexts.
- **One tail is too tight** (1 failure). For `spec_self`'s TPOT on long prose, p99 is 1.18× the p50, under the 1.2× floor. With acceptance at exactly 1, every round does the same work, so a narrow tail is expected there. It is not a sign of warmup leaking into the measurement.

### What disturbed this run

The run took five hours on a machine with no memory to spare, and three things in it are not about the methods:

- **Memory pressure.** The benchmark process held 5.2 GB, and the editor and other apps pushed swap up to its 4 GB limit several times. The worst episode was at about round 80 of long prose. A watchdog was set to stop the run on a GPU out-of-memory error or on critical pressure held for 30 s; it never had to. No GPU error appears in the log. The memory changes that made the run fit are in commits `f0034c8` and `b83858f`.
- **A 40-minute sleep** during long repetitive, around round 70. Python's `perf_counter` does not advance while macOS sleeps, so no sample contains the pause itself, but the rounds right after waking ran on a cold machine.
- **Tails on long prompts are not trustworthy.** The cached baseline's end-to-end p99 is 3.1–3.8× its p50 on long prompts and 1.3–1.7× on short ones. On long repetitive the p99s of the speculative methods reach 15–34 s against p50s under 2 s. The p50s are what this section relies on; the long-prompt p99s mostly measure swap.

The TPOT drift column (median TPOT of the last 3 rounds over the first 3) was meant to show thermal slowdown, which would push it above 1. Many cells read below 1 instead, down to 0.34, which means the first measured rounds were slower. Three rounds per side is a small sample, and the swap episodes fall at different points in different cells, so the column gives no clean thermal signal for this run.

### The 4-gram rate misses repetition in logs

The repetitive category is web-server log lines. They compress best of all three categories (gzip ratio 0.17, against 0.37 for code and 0.53 for prose on long prompts), yet their median repeated whitespace 4-gram rate is 0.00. Every line carries a fresh timestamp and IP, so most whitespace 4-grams include a field that never repeats. The token-level repetition is still there: the prompt-lookup draft's acceptance on these prompts is 0.63, its highest. gzip ratio is the predictability measure that agrees with acceptance here, and the 4-gram rate should be computed over tokens, or with a smaller n, before it is relied on.

<details>
<summary>All columns, every cell</summary>

TTFT is the first-token definition; "1st round" is the first-round definition (see above). All values are in milliseconds. Drift is the TPOT median over the last 3 rounds divided by that over the first 3.

**long / code**: prompt tokens 300–593, p50 449; gzip ratio 0.37, repeated 4-gram rate 0.02

| Method | α | TTFT p50 | TTFT (1st round) p50 | TPOT p50 / p99 | TPOT (1st round) p50 | Draft prefill p50 | E2E p50 / p99 | Speedup vs cached (E2E p50) | TPOT drift last/first |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `no_cache` |  | 655 | 655 | 662 / 1,083 | 662 | – | 21,201 / 34,324 | 0.13× | 0.63 |
| `cached` |  | 689 | 689 | 66.0 / 125 | 66.0 | – | 2,726 / 8,570 | 1.00× | 0.84 |
| `spec_ngram` | 0.19 | 692 | 773 | 44.1 / 69.6 | 41.4 | 0.0 | 2,070 / 3,655 | 1.32× | 1.13 |
| `spec_draft` | 0.56 | 706 | 975 | 53.2 / 97.3 | 43.2 | 80.2 | 2,327 / 4,102 | 1.17× | 0.68 |
| `spec_self` | 1.00 | 682 | 1,741 | 100 / 355 | 65.3 | 683 | 3,788 / 12,759 | 0.72× | 0.80 |
| `hf_assisted` |  | 838 | 838 | 38.3 / 74.2 | 38.3 | – | 2,005 / 3,098 | 1.36× | 0.76 |

**long / prose**: prompt tokens 311–600, p50 448; gzip ratio 0.53, repeated 4-gram rate 0.00

| Method | α | TTFT p50 | TTFT (1st round) p50 | TPOT p50 / p99 | TPOT (1st round) p50 | Draft prefill p50 | E2E p50 / p99 | Speedup vs cached (E2E p50) | TPOT drift last/first |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `no_cache` |  | 638 | 638 | 654 / 975 | 654 | – | 20,955 / 31,121 | 0.13× | 0.79 |
| `cached` |  | 693 | 693 | 65.0 / 97.6 | 65.0 | – | 2,705 / 10,322 | 1.00× | 0.99 |
| `spec_ngram` | 0.05 | 677 | 762 | 62.4 / 146 | 59.9 | 0.0 | 2,553 / 7,429 | 1.06× | 1.03 |
| `spec_draft` | 0.32 | 681 | 933 | 70.4 / 101 | 63.1 | 77.6 | 2,838 / 3,922 | 0.95× | 1.28 |
| `spec_self` | 1.00 | 658 | 1,687 | 97.0 / 115 | 63.9 | 665 | 3,685 / 4,452 | 0.73× | 0.90 |
| `hf_assisted` |  | 814 | 814 | 52.6 / 75.8 | 52.6 | – | 2,407 / 3,610 | 1.12× | 1.08 |

**long / repetitive**: prompt tokens 301–599, p50 445; gzip ratio 0.17, repeated 4-gram rate 0.00

| Method | α | TTFT p50 | TTFT (1st round) p50 | TPOT p50 / p99 | TPOT (1st round) p50 | Draft prefill p50 | E2E p50 / p99 | Speedup vs cached (E2E p50) | TPOT drift last/first |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `no_cache` |  | 658 | 658 | 721 / 1,961 | 721 | – | 23,093 / 61,594 | 0.12× | 0.86 |
| `cached` |  | 705 | 705 | 69.1 / 324 | 69.1 | – | 2,847 / 10,821 | 1.00× | 0.93 |
| `spec_ngram` | 0.63 | 726 | 832 | 24.1 / 453 | 19.8 | 0.0 | 1,446 / 15,100 | 1.97× | 1.25 |
| `spec_draft` | 0.99 | 730 | 1,013 | 40.1 / 482 | 30.9 | 85.5 | 1,973 / 16,341 | 1.44× | 0.85 |
| `spec_self` | 1.00 | 717 | 1,846 | 105 / 260 | 67.5 | 721 | 3,987 / 8,790 | 0.71× | 0.95 |
| `hf_assisted` |  | 921 | 921 | 25.1 / 909 | 25.1 | – | 1,729 / 33,987 | 1.65× | 1.00 |

**short / code**: prompt tokens 18–36, p50 25; gzip ratio 0.77, repeated 4-gram rate 0.00

| Method | α | TTFT p50 | TTFT (1st round) p50 | TPOT p50 / p99 | TPOT (1st round) p50 | Draft prefill p50 | E2E p50 / p99 | Speedup vs cached (E2E p50) | TPOT drift last/first |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `no_cache` |  | 78.7 | 78.7 | 96.8 / 139 | 96.8 | – | 3,070 / 4,385 | 0.59× | 1.27 |
| `cached` |  | 84.3 | 84.3 | 55.2 / 68.1 | 55.2 | – | 1,808 / 2,332 | 1.00× | 1.02 |
| `spec_ngram` | 0.06 | 83.4 | 144 | 51.4 / 84.9 | 48.9 | 0.0 | 1,687 / 2,732 | 1.07× | 1.09 |
| `spec_draft` | 0.64 | 85.5 | 255 | 41.2 / 92.2 | 35.3 | 23.1 | 1,410 / 2,163 | 1.28× | 1.10 |
| `spec_self` | 1.00 | 83.6 | 447 | 66.8 / 122 | 54.8 | 75.4 | 2,164 / 4,985 | 0.84× | 1.09 |
| `hf_assisted` |  | 190 | 190 | 32.3 / 80.7 | 32.3 | – | 1,207 / 2,757 | 1.50× | 0.97 |

**short / prose**: prompt tokens 9–22, p50 15; gzip ratio 0.74, repeated 4-gram rate 0.00

| Method | α | TTFT p50 | TTFT (1st round) p50 | TPOT p50 / p99 | TPOT (1st round) p50 | Draft prefill p50 | E2E p50 / p99 | Speedup vs cached (E2E p50) | TPOT drift last/first |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `no_cache` |  | 74.6 | 74.6 | 88.7 / 128 | 88.7 | – | 2,819 / 4,097 | 0.64× | 0.81 |
| `cached` |  | 80.3 | 80.3 | 55.1 / 81.1 | 55.1 | – | 1,796 / 2,751 | 1.00× | 0.99 |
| `spec_ngram` | 0.04 | 79.1 | 139 | 56.8 / 108 | 54.6 | 0.0 | 1,832 / 3,042 | 0.98× | 0.97 |
| `spec_draft` | 0.37 | 82.7 | 256 | 56.8 / 87.4 | 51.6 | 22.1 | 1,837 / 2,927 | 0.98× | 0.99 |
| `spec_self` | 1.00 | 79.9 | 437 | 65.9 / 162 | 54.3 | 67.3 | 2,130 / 8,445 | 0.84× | 0.99 |
| `hf_assisted` |  | 186 | 186 | 48.5 / 76.9 | 48.5 | – | 1,664 / 10,695 | 1.08× | 0.91 |

**short / repetitive**: prompt tokens 37–167, p50 80; gzip ratio 0.44, repeated 4-gram rate 0.06

| Method | α | TTFT p50 | TTFT (1st round) p50 | TPOT p50 / p99 | TPOT (1st round) p50 | Draft prefill p50 | E2E p50 / p99 | Speedup vs cached (E2E p50) | TPOT drift last/first |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `no_cache` |  | 177 | 177 | 192 / 339 | 192 | – | 6,133 / 10,950 | 0.32× | 1.71 |
| `cached` |  | 185 | 185 | 56.7 / 103 | 56.7 | – | 1,940 / 3,310 | 1.00× | 0.69 |
| `spec_ngram` | 0.32 | 177 | 245 | 28.8 / 85.0 | 26.8 | 0.0 | 1,090 / 2,797 | 1.78× | 0.34 |
| `spec_draft` | 0.69 | 186 | 383 | 39.9 / 68.0 | 33.3 | 30.5 | 1,487 / 2,374 | 1.30× | 0.60 |
| `spec_self` | 1.00 | 180 | 653 | 73.6 / 105 | 56.7 | 174 | 2,429 / 3,449 | 0.80× | 0.84 |
| `hf_assisted` |  | 276 | 276 | 29.7 / 50.2 | 29.7 | – | 1,256 / 2,037 | 1.54× | 0.47 |

</details>

## Correctness

### Greedy identity

At temperature 0 the speculative loop must emit exactly the target's greedy sequence.

| Setup | Prompts | Tokens | Result |
|:--|--:|--:|:--|
| Tiny random Llama (2 layers, fp64), 1-layer early-exit draft, k = 4 | 200 | 24 | 200/200 identical (also batched at 8 and 25, and k = 1, 2, 3, 8) |
| SmolLM2-360M target / SmolLM2-135M draft, fp64, CPU | 50 | 16 | 50/50 identical |
| SmolLM2-1.7B target / SmolLM2-135M draft, bf16 weights, fp32 logits, Apple M3 (MPS) | 200 | 32 | 194/200 identical; all 6 flips are rounding (below) |
| Same, bf16 logits (before the fix below) | 200 | 32 | 187/200 identical |

The real-weight exact check uses the 360M target because the 1.7B needs 14 GiB in fp64. An earlier 32-token fp64 run was stopped at 78 prompts, with 78/78 identical, when the prompt count was cut to 50.

### bf16 flips are rounding, not the algorithm

In a 6-prompt smoke run on the 1.7B/135M pair in bf16, one prose prompt diverged at token 26. The reference's top two logits there are `' went'` 18.625 and `' then'` 18.5. Between 16 and 32 the bf16 grid spacing is 0.125, so the two candidates are one grid step apart. The verify pass runs k + 1 tokens through one matmul and plain decoding runs one token at a time. Those shapes round differently, and a one-step tie can come out either way. The same prompt is identical between cached and uncached plain decoding, and the n-gram draft diverges at the same position, which rules out the draft model.

An accept step that is too lenient would also flip near-ties, so a small logit gap does not excuse a flip by itself. Three conditions are asserted on the 200-prompt run:

- each flip's top-2 gap is at most two grid steps of the model's compute dtype;
- fewer than 5% of prompts flip;
- the flips do not favour the draft. Rounding picks a side of the tie without reference to the draft, so the emitted token is the draft's at most about half the time. In a simulated-rounding control on the tiny model the figure was 22–27%. A lenient accept emits the draft's token every time. The test uses a one-sided binomial test against 1/2. A known-bad verifier that accepts any draft within 0.001 of the max logit is caught by it: every flip emits the draft's token.

With bf16 logits the first run missed the second condition: 13 of 200 prompts flipped (6.5%). Seven of the 13 flips sat at an exact tie: the two candidates rounded to the same bf16 value, and argmax broke the tie differently in the two passes. The output projection now runs in fp32 for every method, the baselines included. The rerun flipped 6 of 200 (3.0%), with gaps between 0.001 and 0.071 logits (at most 1.14 grid steps) and no exact ties. One of the 6 flips emitted the draft's token (one-sided p = 0.98). What remains comes from bf16 hidden states, which fp32 logits do not change.

### Distribution preservation

On a 32-token vocabulary with k = 4 and 100,000 draws per side, speculative sampling matches plain sampling from the target within sampling noise. At position 0, total-variation distance is 0.0072 against an expected noise level of 0.0085, with a goodness-of-fit p of 0.134. Resampling from `p_target` instead of the residual `norm(max(0, p_target - p_draft))` gives a TV of 0.1684 and p = 0, so it is rejected. On a 5-token toy vocabulary, the accept and residual probabilities are asserted equal to hand-computed fractions under a shared random stream.

### KV cache equivalence

Cached decoding matches a full recomputation at every step to within 1e-10 in fp64. This holds for one sequence, for a padded batch of mixed lengths, and after a rollback. On real SmolLM2-135M weights in fp64 the maximum difference is 2.2e-14. In fp32 on the M3 it is 1.12e-4, which misses the 1e-4 criterion. The cause is that 1-token and n-token matmuls round differently, not the cache.
