import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
# from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration
class Collator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        # 👇 强制读取，不管有没有，只读取存在的
        has_coarse = all("coarse_labels" in d for d in batch)
        coarse_texts = None
        if has_coarse:
            coarse_texts = [d["coarse_labels"] for d in batch]

        # 编码输入
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True
        )

        # 编码细标签
        labels = self.tokenizer(
            label_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        )
        inputs["labels"] = labels["input_ids"]
        inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100

        # 👇 粗标签：只有全部都有才传
        if has_coarse and coarse_texts is not None:
            co_lab = self.tokenizer(
                coarse_texts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True
            )
            inputs["coarse_labels"] = co_lab["input_ids"]
            inputs["coarse_labels"][inputs["coarse_labels"] == self.tokenizer.pad_token_id] = -100

        return inputs

# class Collator(object):

#     def __init__(self, args, tokenizer):
#         self.args = args
#         self.only_train_response = args.only_train_response
#         self.tokenizer = tokenizer
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = 0
#         # print(self.tokenizer.model_max_length)

#     def __call__(self, batch):

#         input_texts = [d["input_ids"] for d in batch]
#         label_texts = [d["labels"] for d in batch]
#         has_coarse = all("coarse_labels" in d and d["coarse_labels"] is not None for d in batch)
#         coarse_texts = [d["coarse_labels"] for d in batch] if has_coarse else None

#         inputs = self.tokenizer(input_texts,
#                                 return_tensors="pt",
#                                 padding="longest",
#                                 max_length=self.tokenizer.model_max_length,
#                                 truncation=True,
#                                 return_attention_mask=True)

#         labels = self.tokenizer(label_texts,
#                                 return_tensors="pt",
#                                 padding="longest",
#                                 max_length=self.tokenizer.model_max_length,
#                                 truncation=True,
#                                 return_attention_mask=True)
#         coarse_labels = self.tokenizer(coarse_texts,
#                                 return_tensors="pt",
#                                 padding="longest",
#                                 max_length=self.tokenizer.model_max_length,
#                                 truncation=True,
#                                 return_attention_mask=True)
#         inputs['labels'] = labels['input_ids']
#         inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
#         if has_coarse and coarse_texts is not None:
#             coarse_labels = self.tokenizer(coarse_texts,
#                                     return_tensors="pt",
#                                     padding="longest",
#                                     max_length=self.tokenizer.model_max_length,
#                                     truncation=True)
#             inputs["coarse_labels"] = coarse_labels["input_ids"]
#             inputs["coarse_labels"][inputs["coarse_labels"] == self.tokenizer.pad_token_id] = -100

#         return inputs



class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)

