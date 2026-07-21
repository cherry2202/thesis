# HoloRec-TIGER-Progressive

This folder contains an alternative design for the HoloRec-TIGER coarse/fine recommendation model.

## Motivation

The original `8 -> 32` design uses an 8-dimensional coarse code as a hard bottleneck before predicting the 32-dimensional fine code. If the 8-code space is too compressed or poorly aligned with downstream sequential behavior, the coarse latent can inject misleading information into every fine prediction step.

This variant replaces the single coarse bottleneck with **progressive prefix supervision**:

1. Keep the 32-token fine code as the final target.
2. Derive several prefix-level targets from the same fine code, for example levels `8,16,24`.
3. Add one auxiliary head per prefix level.
4. Predict fine tokens with a gated mixture of prefix-level latent embeddings.
5. Weight auxiliary prefix losses lightly so they regularize the fine objective without becoming a new hard bottleneck.

The key difference is that the model no longer depends on a separate 8-code file. Coarse supervision is derived from the fine code prefixes, so the hierarchy is guaranteed to be consistent with the final item code.

## Files

- `data.py`: dataset loader that reads the interaction file and the fine index file only.
- `collator.py`: builds fine labels and prefix-level labels.
- `modeling_progressive.py`: T5-based model with progressive prefix heads and adaptive latent fusion.
- `finetune.py`: training entry point.
- `run_train.sh`: example training script.

## Recommended starting configuration

```bash
bash HoloRec-TIGER-Progressive/run_train.sh
```

For a 32-token fine code, start with:

- `--prefix_levels 8,16,24`
- `--prefix_loss_weights 0.2,0.15,0.1`
- `--fine_loss_weight 1.0`
- `--prefix_latent_weight 0.5`

If the prefix losses dominate, lower `--prefix_latent_weight` or `--prefix_loss_weights`. If the model overfits fine codes without learning structure, increase the first prefix loss slightly.
