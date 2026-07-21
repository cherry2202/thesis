import torch


class ProgressiveCollator:
    """Builds fine labels and prefix-level labels for progressive training."""

    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.prefix_levels = [int(x) for x in args.prefix_levels.split(",") if x]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def _token_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(ids, int):
            ids = [ids]
        unk = self.tokenizer.unk_token_id
        if unk is not None and any(token_id == unk for token_id in ids):
            bad = [tok for tok, token_id in zip(tokens, ids) if token_id == unk]
            raise ValueError(f"Unknown fine-code tokens: {bad}")
        return ids

    def _pad(self, sequences, pad_value=-100):
        max_len = max(len(seq) for seq in sequences)
        return torch.tensor([seq + [pad_value] * (max_len - len(seq)) for seq in sequences], dtype=torch.long)

    def __call__(self, batch):
        input_texts = [row["input_ids"] for row in batch]
        fine_id_rows = [self._token_ids(row["fine_codes"]) for row in batch]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        fine_labels = self._pad(fine_id_rows)
        prefix_labels = {}
        for level in self.prefix_levels:
            prefix_labels[str(level)] = self._pad([ids[: min(level, len(ids))] for ids in fine_id_rows])

        inputs["labels"] = fine_labels
        inputs["fine_labels"] = fine_labels
        inputs["prefix_labels"] = prefix_labels
        return inputs
