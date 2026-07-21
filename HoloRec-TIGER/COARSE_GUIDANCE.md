# Coarse-guided inference for the 8-dim/32-dim RQ-VAE setup

## Your current setting

Each item has two independent RQ-VAE codes:

- **Coarse code**: run RQ-VAE on the first 8 dimensions of the item semantic vector. The output is a 4-token code, for example `[c1, c2, c3, c4]`.
- **Fine code**: run another RQ-VAE on the full 32-dimensional semantic vector. The output is another 4-token code, for example `[f1, f2, f3, f4]`.

The coarse code is not a prefix of the fine code. It is a separate view of the same item. Therefore it should be used at inference as an **item-level prior** or **candidate filter**, not by slicing the fine code.

## Implemented inference strategy

The updated `test.py` keeps the original fine-code beam search, but adds optional item-aware coarse guidance:

1. Build a lookup from every partial fine prefix to the coarse tokens of items that share that fine prefix.
2. During each fine-token beam expansion, predict the model's coarse-token distribution for the same step.
3. For each candidate fine prefix, look up which coarse tokens are compatible with real items under that prefix.
4. Add `coarse_guidance_weight * max log P(compatible coarse token)` to the candidate fine beam score.

In short:

```text
score = fine_beam_score + fine_logprob + lambda * coarse_compatibility_logprob
```

where `lambda` is `--coarse_guidance_weight`.

This lets the coarse RQ-VAE code help choose between fine-code candidates without forcing the generated result to be the coarse code itself.

## How to run

Start with a small guidance weight:

```bash
torchrun --nproc_per_node=8 --master_port=2214 test.py \
  --ckpt_path ./ckpt/Instruments-ceshi/checkpoint-16254 \
  --dataset Instruments \
  --data_path ../data \
  --results_file ./results/Instruments/coarse-guided.json \
  --test_batch_size 32 \
  --num_beams 20 \
  --index_file .tw32.json \
  --coarse_index_file .tw8.json \
  --coarse_guidance_weight 0.2
```

Recommended sweep:

```text
0.0, 0.05, 0.1, 0.2, 0.3, 0.5
```

If performance drops, the coarse code is probably noisy for ranking and should be used only as a recall-stage shortlist with a larger fine beam, or retrained with behavior-aware clustering.
