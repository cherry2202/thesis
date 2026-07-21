import json
import os
import numpy as np
from torch.utils.data import Dataset


class SeqRecDataset(Dataset):
    """Sequential recommendation dataset for progressive-prefix TIGER.

    The dataset reads the original interaction file and the fine-code index file.
    Prefix-level targets are derived later by the collator from the fine code, so
    no separate coarse-code file is required.
    """

    def __init__(self, args, mode="train", sample_num=-1):
        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.index_file = args.index_file
        self.max_his_len = args.max_his_len
        self.add_prefix = args.add_prefix
        self.sample_num = sample_num
        self.mode = mode

        self._load_data()
        self._remap_items()
        self.inter_data = self._process_data()

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r", encoding="utf-8") as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r", encoding="utf-8") as f:
            self.indices = json.load(f)

    def _remap_items(self):
        self.remapped_inters = {}
        for uid, items in self.inters.items():
            self.remapped_inters[uid] = ["".join(self.indices[str(item)]) for item in items]

    def get_new_tokens(self):
        tokens = set()
        for code in self.indices.values():
            tokens.update(code)
        return sorted(tokens)

    def get_all_items(self):
        return ["".join(code) for code in self.indices.values()]

    def _build_history(self, items):
        if self.max_his_len > 0:
            items = items[-self.max_his_len:]
        if self.add_prefix:
            items = [f"{idx + 1}. {item}" for idx, item in enumerate(items)]
        return "".join(items)

    def _build_one_data(self, target_item_id, history_items):
        return {
            "input_ids": self._build_history(history_items),
            "fine_codes": self.indices[str(target_item_id)],
        }

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            raw_items = self.inters[uid][:-2]
            fine_items = self.remapped_inters[uid][:-2]
            for pos in range(1, len(raw_items)):
                inter_data.append(self._build_one_data(raw_items[pos], fine_items[:pos]))
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.inters:
            raw_items = self.inters[uid]
            fine_items = self.remapped_inters[uid]
            inter_data.append(self._build_one_data(raw_items[-2], fine_items[:-2]))
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.inters:
            raw_items = self.inters[uid]
            fine_items = self.remapped_inters[uid]
            inter_data.append(self._build_one_data(raw_items[-1], fine_items[:-1]))
        if self.sample_num > 0:
            sample_idx = np.random.choice(range(len(inter_data)), self.sample_num, replace=False)
            inter_data = np.array(inter_data)[sample_idx].tolist()
        return inter_data

    def _process_data(self):
        if self.mode == "train":
            return self._process_train_data()
        if self.mode == "valid":
            return self._process_valid_data()
        if self.mode == "test":
            return self._process_test_data()
        raise NotImplementedError(f"Unknown mode: {self.mode}")

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        return self.inter_data[index]
