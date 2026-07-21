from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


class CoarseFineIndex:
    """Index connecting separate coarse and fine RQ-VAE item codes.

    A fine prefix may correspond to many real items. Those items each have a
    coarse token at the same code step. This index lets the decoder ask:
    "if I extend the fine prefix this way, which coarse tokens are compatible
    with actual items?"
    """

    def __init__(self, test_data, tokenizer):
        self.tokenizer = tokenizer
        self.fine_sequences = []
        self.coarse_sequences = []
        self.all_fine_texts = []
        self.prefix_to_coarse_ids: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
        self._build(test_data)

    def _convert_tokens(self, tokens: Sequence[str]) -> List[int]:
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(ids, int):
            ids = [ids]
        return list(ids)

    def _build(self, test_data):
        prefix_to_coarse_sets = defaultdict(set)

        for item_id, fine_tokens in test_data.indices.items():
            coarse_tokens = test_data.coarse_indices[str(item_id)]
            if len(fine_tokens) != len(coarse_tokens):
                raise ValueError(
                    f"item {item_id} has fine/coarse length mismatch: "
                    f"{len(fine_tokens)} vs {len(coarse_tokens)}"
                )

            fine_ids = self._convert_tokens(fine_tokens)
            coarse_ids = self._convert_tokens(coarse_tokens)
            self.fine_sequences.append(fine_ids)
            self.coarse_sequences.append(coarse_ids)
            self.all_fine_texts.append("".join(fine_tokens))

            for step_idx in range(len(fine_ids)):
                fine_prefix = tuple(fine_ids[: step_idx + 1])
                prefix_to_coarse_sets[(step_idx, fine_prefix)].add(coarse_ids[step_idx])

        self.prefix_to_coarse_ids = {
            key: sorted(value) for key, value in prefix_to_coarse_sets.items()
        }

    def compatible_coarse_ids(self, step_idx: int, fine_prefix: Iterable[int]) -> List[int]:
        return self.prefix_to_coarse_ids.get((step_idx, tuple(fine_prefix)), [])
