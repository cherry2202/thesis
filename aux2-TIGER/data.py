import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
import numpy as np
from transformers import T5Tokenizer


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.coarse_index_file = args.coarse_index_file
        self.mid_index_file = args.mid_index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None
        # ================新增：粗粒度字典==========
        self.item2coarse = None
        self.item2mid = None
        
    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        # =================新增：加载粗粒度编码文件=============
        coarse_file = os.path.join(self.data_path, self.dataset + self.coarse_index_file)
        if os.path.exists(coarse_file):
            with open(coarse_file, 'r') as f:
                self.item2coarse = json.load(f)
            print("已加载粗粒度SID")
        # =================新增：加载中粒度编码文件=============
        mid_file = os.path.join(self.data_path, self.dataset + self.mid_index_file)
        if os.path.exists(mid_file):
            with open(mid_file, 'r') as f:
                self.item2mid = json.load(f)
            print("已加载中粒度SID")

    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items       
    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num


        # load data
        self._load_data()
        
        # ========== 新增：过滤无效 ID ==========
        self._filter_invalid_items()
        
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        # =================新增：加载粗粒度==================
        coarse_file = os.path.join(self.data_path, self.dataset + self.coarse_index_file)
        if os.path.exists(coarse_file):
            with open(coarse_file, 'r') as f:
                self.item2coarse = json.load(f)
            print("已加载粗粒度SID")        
        # =================新增：加载中粒度==================
        mid_file = os.path.join(self.data_path, self.dataset + self.mid_index_file)
        if os.path.exists(mid_file):
             with open(mid_file, 'r') as f:
                self.item2mid = json.load(f)
             print("已加载中粒度SID")
             
    # （保留原有所有方法，新增以下方法）
    def _filter_invalid_items(self):
        """
        过滤 self.inters 中不在 self.indices 里的 item ID
        因为你已经删了 index 文件中不需要的 ID，所以只保留 index 中存在的 ID
        """
        # 提取 index 文件中的有效 ID（转为整数，方便对比）
        valid_item_ids = set(int(k) for k in self.indices.keys())
        
        # 过滤每个用户的交互 item
        filtered_inters = {}
        for uid, items in self.inters.items():
            # 只保留有效 ID 的 item
            filtered_items = [i for i in items if i in valid_item_ids]
            # 只保留有有效交互的用户（避免空列表）
            if len(filtered_items) >= 2:  # 至少保留2个item（满足train/valid/test的切片逻辑）
                filtered_inters[uid] = filtered_items
        
        # 替换原 inters 为过滤后的结果
        self.inters = filtered_inters
        print(f"过滤后保留 {len(self.inters)} 个用户，删除了无效 item ID")
    
    def _remap_items(self):

        self.remapped_inters = dict()
        # ==============新增：粗粒度映射===============
        self.remapped_coarse_inters = dict()
        self.remapped_mid_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items
            # =================coarse===============
            if self.item2coarse is not None:
                coarse_items = ["".join(self.item2coarse[str(i)]) for i in items]
                self.remapped_coarse_inters[uid] = coarse_items
            # =================mid===============
            if self.item2mid is not None:
                mid_items = ["".join(self.item2mid[str(i)]) for i in items]
                self.remapped_mid_inters[uid] = mid_items
        # 新增日志：验证 remap 结果
        print(f"完成 item ID 映射，共处理 {len(self.remapped_inters)} 个用户")

    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            coarse_items = self.remapped_coarse_inters[uid][:-2]
            mid_items = self.remapped_mid_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                # one_data["user"] = uid
                one_data["item"] = items[i]
                one_data["coarse_item"] = coarse_items[i]  # ==============新增：粗粒度标签===============
                one_data["mid_item"] = mid_items[i]  # ==============新增：中粒度标签===============
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                one_data["inters"] = "".join(history)
                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            coarse_items = self.remapped_coarse_inters[uid]
            mid_items = self.remapped_mid_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            one_data["coarse_item"] = coarse_items[-2]  # ==============新增：粗粒度标签===============
            one_data["mid_item"] = mid_items[-2]  # ==============新增：中粒度标签===============
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            # if uid not in cold_user:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = "".join(history)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data
    
    def _process_test_data_ids(self):

        inter_data = []
        for uid in self.inters:
            # if uid not in cold_user:
            items = self.inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data       
    

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    # def __getitem__(self, index):


    #     d = self.inter_data[index]
    #     # ====================返回粗细=================
    #     return dict(input_ids=d["inters"], labels=d["item"], coarse_labels=d.get("coarse_item", None))
    def __getitem__(self, index):
        d = self.inter_data[index]
        
        # 基础字段永远有
        out = {
            "input_ids": d["inters"],
            "labels": d["item"]
        }

        # 👇 只有训练集才加 coarse_labels！
        # 👇 验证/测试 绝对不加！
        if self.mode == "train" or self.mode == "valid":
            out["coarse_labels"] = d["coarse_item"]
            out["mid_labels"] = d["mid_item"]

        return out