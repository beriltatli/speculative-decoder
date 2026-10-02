# Can a small draft model make a large one generate faster without changing a single output token?

**Status: Phases 0–3 of 6 complete (KV cache, accept-reject, draft-verify loop). No speed measurements yet; those start in Phase 4. Two real-weight identity runs are in progress and marked as such below.**

## Correctness

### Greedy identity

At temperature 0 the speculative loop must emit exactly the target's greedy sequence.

| Setup | Prompts | Tokens | Result |
|:--|--:|--:|:--|
| Tiny random Llama (2 layers, fp64), 1-layer early-exit draft, k = 4 | 200 | 24 | 200/200 identical (also batched at 8 and 25, and k = 1, 2, 3, 8) |
| SmolLM2-360M target / SmolLM2-135M draft, fp64, CPU | 50 | 16 | running |
| SmolLM2-1.7B target / SmolLM2-135M draft, bf16, Apple M3 (MPS) | 200 | 32 | running |

The real-weight exact check uses the 360M target because the 1.7B needs 14 GiB in fp64. An earlier 32-token fp64 run was stopped at 78 prompts, with 78/78 identical, when the prompt count was cut to 50.

### bf16 flips are rounding, not the algorithm

In a 6-prompt smoke run on the 1.7B/135M pair in bf16, one prose prompt diverged at token 26. The reference's top two logits there are `' went'` 18.625 and `' then'` 18.5. Between 16 and 32 the bf16 grid spacing is 0.125, so the two candidates are one grid step apart. The verify pass runs k + 1 tokens through one matmul and plain decoding runs one token at a time. Those shapes round differently, and a one-step tie can come out either way. The same prompt is identical between cached and uncached plain decoding, and the n-gram draft diverges at the same position, which rules out the draft model.

An accept step that is too lenient would also flip near-ties, so a small logit gap does not excuse a flip by itself. On the 200-prompt bf16 run, three conditions are asserted:

- each flip's top-2 gap is at most two grid steps;
- fewer than 5% of prompts flip;
- the flips do not favour the draft. Rounding picks a side of the tie without reference to the draft, so the emitted token is the draft's at most about half the time. In a simulated-rounding control on the tiny model the figure was 22–27%. A lenient accept emits the draft's token every time. The test uses a one-sided binomial test against 1/2. A known-bad verifier that accepts any draft within 0.001 of the max logit is caught by it: every flip emits the draft's token.

### Distribution preservation

On a 32-token vocabulary with k = 4 and 100,000 draws per side, speculative sampling matches plain sampling from the target within sampling noise. At position 0, total-variation distance is 0.0072 against an expected noise level of 0.0085, with a goodness-of-fit p of 0.134. Resampling from `p_target` instead of the residual `norm(max(0, p_target - p_draft))` gives a TV of 0.1684 and p = 0, so it is rejected. On a 5-token toy vocabulary, the accept and residual probabilities are asserted equal to hand-computed fractions under a shared random stream.

### KV cache equivalence

Cached decoding matches a full recomputation at every step to within 1e-10 in fp64. This holds for one sequence, for a padded batch of mixed lengths, and after a rollback. On real SmolLM2-135M weights in fp64 the maximum difference is 2.2e-14. In fp32 on the M3 it is 1.12e-4, which misses the 1e-4 criterion. The cause is that 1-token and n-token matmuls round differently, not the cache.
