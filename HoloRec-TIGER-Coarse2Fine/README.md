# HoloRec-TIGER-Coarse2Fine

This folder implements a stronger inference-time use of your two RQ-VAE item codes.

## Is the previous soft-guidance scheme feasible?

Yes, but it is only a weak reranker. It adds a coarse-code compatibility score to each fine-code beam expansion, but the search space is still mainly controlled by the fine trie. If the fine beam prunes a useful item early, a later coarse score cannot recover it.

## Better scheme: coarse-filtered fine beam search

Your setting has two independent item codes:

- `coarse_code`: 4 tokens from RQ-VAE over the first 8 semantic-vector dimensions.
- `fine_code`: 4 tokens from another RQ-VAE over the full 32-dimensional vector.

The coarse code should be used as a **dynamic item prior** during fine decoding:

1. At each decoding step, use the model's `coarse_head` to predict a distribution over coarse tokens conditioned on the user history and current fine prefix.
2. For each candidate fine-token expansion, use the item index to find which real items have that new fine prefix and which coarse token those items have at the same step.
3. Keep candidates whose compatible coarse tokens are in the top-K coarse tokens, or keep them with a penalty in `hybrid` mode.
4. Rank the remaining candidates with:

```text
score = previous_beam_score
      + fine_logprob
      + coarse_guidance_weight * max_logprob(compatible coarse tokens)
      - coarse_mismatch_penalty_if_needed
```

This is more useful than pure soft guidance because coarse codes can influence **which branches survive** in beam search, not just the final reranking.

## Files

- `coarse_index.py`: builds the fine-prefix to compatible-coarse-token index.
- `coarse2fine_test.py`: standalone evaluation script with coarse-filtered fine beam search.
- `run_test.sh`: example launch command.

## Recommended sweep

Start with hybrid filtering because it is robust when the coarse code is noisy:

```bash
bash HoloRec-TIGER-Coarse2Fine/run_test.sh
```

Suggested values:

```text
--coarse_filter_mode hybrid
--coarse_topk_per_step 8,16,32
--coarse_guidance_weight 0.1,0.2,0.3
--coarse_mismatch_penalty 0.5,1.0,2.0
```

If coarse quality is high, try `--coarse_filter_mode hard`. If performance drops, use `soft` or increase `--coarse_topk_per_step`.
