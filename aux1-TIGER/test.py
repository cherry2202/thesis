# import argparse
# import json
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# import sys
# from typing import List

# import torch
# import transformers
# # from peft import PeftModel
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration

# from utils import *
# from collator import TestCollator
# from evaluate import get_topk_results, get_metrics_results
# from generation_trie import Trie


# def test(args):

#     set_seed(args.seed)
#     print(vars(args))

#     device_map = {"": args.gpu_id}
#     device = torch.device("cuda",args.gpu_id)

#     config = T5Config.from_pretrained("t5-small")
#     tokenizer = T5Tokenizer.from_pretrained(
#         "t5-small",
#         model_max_length=512,
#     )
#     train_data, valid_data = load_datasets(args)
#     add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
#     config.vocab_size = len(tokenizer)

#     print("add {} new token.".format(add_num))
#     print("data num:", len(train_data))

#     # tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path)
#     model = T5ForConditionalGeneration.from_pretrained(
#         args.ckpt_path,
#         low_cpu_mem_usage=True,
#         device_map=device_map,
#     )

#     prompt_ids = [0]

#     test_data = load_test_dataset(args)

#     collator = TestCollator(args, tokenizer)
#     all_items = test_data.get_all_items()


#     candidate_trie = Trie(
#         [
#             [0] + tokenizer.encode(candidate)
#             for candidate in all_items
#         ]
#     )
#     prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

#     test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
#                              shuffle=True, num_workers=4, pin_memory=True)


#     print("data num:", len(test_data))

#     model.eval()

#     metrics = args.metrics.split(",")
#     all_prompt_results = []
#     with torch.no_grad():
#         for prompt_id in prompt_ids:

            
#             test_loader.dataset.set_prompt(prompt_id)
#             metrics_results = {}
#             total = 0

#             for step, batch in enumerate(tqdm(test_loader)):
#                 inputs = batch[0].to(device)
#                 targets = batch[1]
#                 total += len(targets)
#                 if step == 0:
#                     print(inputs)
#                     print(targets)

#                 output = model.generate(
#                     input_ids=inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"],
#                     max_new_tokens=10,
#                     # max_length=10,
#                     prefix_allowed_tokens_fn=prefix_allowed_tokens,
#                     num_beams=args.num_beams,
#                     num_return_sequences=args.num_beams,
#                     output_scores=True,
#                     return_dict_in_generate=True,
#                     early_stopping=True,
#                 )
#                 output_ids = output["sequences"]
#                 scores = output["sequences_scores"]

#                 output = tokenizer.batch_decode(
#                     output_ids, skip_special_tokens=True
#                 )

#                 topk_res = get_topk_results(output,scores,targets,args.num_beams,
#                                             all_items=all_items if args.filter_items else None)

#                 batch_metrics_res = get_metrics_results(topk_res, metrics)
#                 # print(batch_metrics_res)

#                 for m, res in batch_metrics_res.items():
#                     if m not in metrics_results:
#                         metrics_results[m] = res
#                     else:
#                         metrics_results[m] += res

#                 # if (step+1)%10 == 0:
#                 temp={}
#                 for m in metrics_results:
#                     temp[m] = metrics_results[m] / total
#                 print(temp)

#             for m in metrics_results:
#                 metrics_results[m] = metrics_results[m] / total
#             all_prompt_results.append(metrics_results)
#             print("======================================================")
#             print("Prompt {} results: ".format(prompt_id), metrics_results)
#             print("======================================================")
#             print("")

#     mean_results = {}
#     min_results = {}
#     max_results = {}

#     for m in metrics:
#         all_res = [_[m] for _ in all_prompt_results]
#         mean_results[m] = sum(all_res)/len(all_res)
#         min_results[m] = min(all_res)
#         max_results[m] = max(all_res)

#     print("======================================================")
#     print("Mean results: ", mean_results)
#     print("Min results: ", min_results)
#     print("Max results: ", max_results)
#     print("======================================================")


#     save_data={}
#     save_data["test_prompt_ids"] = args.test_prompt_ids
#     save_data["mean_results"] = mean_results
#     save_data["min_results"] = min_results
#     save_data["max_results"] = max_results
#     save_data["all_prompt_results"] = all_prompt_results

#     with open(args.results_file, "w") as f:
#         json.dump(save_data, f, indent=4)



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="LLMRec_test")
#     parser = parse_global_args(parser)
#     parser = parse_dataset_args(parser)
#     parser = parse_test_args(parser)

#     args = parser.parse_args()

#     test(args)


import argparse
import json
import os
import sys
from typing import List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie

from modeling_letter import LETTER

def init_distributed():
    """
    torchrun 会设置 RANK/WORLD_SIZE/LOCAL_RANK
    返回: (is_dist, local_rank, rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        return True, local_rank, rank, world_size
    return False, 0, 0, 1


def is_rank0(is_dist: bool, rank: int) -> bool:
    return (not is_dist) or rank == 0


def parse_prompt_ids(test_prompt_ids) -> List[int]:
    """
    兼容 args.test_prompt_ids 为 int / list / "0,1,2" 字符串等情况
    """
    if test_prompt_ids is None:
        return [0]
    if isinstance(test_prompt_ids, int):
        return [test_prompt_ids]
    if isinstance(test_prompt_ids, list):
        return [int(x) for x in test_prompt_ids]
    if isinstance(test_prompt_ids, str):
        s = test_prompt_ids.strip()
        if not s:
            return [0]
        return [int(x) for x in s.split(",")]
    return [0]


def all_reduce_metrics(metrics_results: dict, total: int, device: torch.device, is_dist: bool):
    """
    把各 rank 的累计指标值与 total 样本数做 all_reduce 求和，然后除以全局 total
    """
    if not is_dist:
        # 单卡：直接归一化
        if total == 0:
            return {k: 0.0 for k in metrics_results}
        return {k: float(v) / float(total) for k, v in metrics_results.items()}

    # 分布式：all_reduce total
    total_tensor = torch.tensor([total], device=device, dtype=torch.long)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    total_all = int(total_tensor.item())

    # all_reduce 每个指标的分子
    out = {}
    for k, v in metrics_results.items():
        t = torch.tensor([float(v)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[k] = (t.item() / total_all) if total_all > 0 else 0.0

    return out


def test(args):
    is_dist, local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)

    # 为了可复现且避免各 rank 完全相同随机序列（比如 sampler shuffle=True 时）
    base_seed = getattr(args, "seed", 42)
    set_seed(base_seed + rank)

    if is_rank0(is_dist, rank):
        print(vars(args))
        print(f"[DDP] is_dist={is_dist}, world_size={world_size}")

    # ===== Tokenizer / Config =====
    config = T5Config.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained(
        "t5-small",
        model_max_length=512,
    )

    # 载入数据（保持你原来的逻辑）
    # train_data, valid_data = load_datasets(args)
    # add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    # config.vocab_size = len(tokenizer)
    test_data = load_test_dataset(args)
    # 从测试集拿新词表，完全不碰训练集
    add_num = tokenizer.add_tokens(test_data.get_new_tokens())
    config.vocab_size = len(tokenizer)

    if is_rank0(is_dist, rank):
        print(f"add {add_num} new token.")
        # print("train data num:", len(train_data))

    # ===== Model =====
    # DDP 数据并行：每个进程一份模型，放在自己的 GPU 上
    model = T5ForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        low_cpu_mem_usage=True,
    )
    # model = LETTER.from_pretrained(
    #     args.ckpt_path,
    #     low_cpu_mem_usage=True,
    # )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    # prompt_ids
    prompt_ids = parse_prompt_ids(getattr(args, "test_prompt_ids", 0))

    # ===== Test Dataset =====
    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    # Trie / prefix constraint（每个 rank 各自构建一份即可）
    candidate_trie = Trie([[0] + tokenizer.encode(candidate) for candidate in all_items])
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    # DistributedSampler：确保每张卡拿不同数据
    sampler = DistributedSampler(
        test_data,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,  # 测试建议不 shuffle，保证确定性；如你必须 shuffle，可改 True 并 set_epoch
        drop_last=False
    ) if is_dist else None

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=(sampler is None),   # 单卡时可 shuffle；分布式由 sampler 控制
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    if is_rank0(is_dist, rank):
        print("test data num:", len(test_data))

    metrics = args.metrics.split(",")
    all_prompt_results = []

    with torch.no_grad():
        for prompt_id in prompt_ids:
            test_loader.dataset.set_prompt(prompt_id)

            # 若 sampler shuffle=True，可每个 prompt 调一次 set_epoch 保证一致 shuffle
            if sampler is not None and getattr(sampler, "shuffle", False):
                sampler.set_epoch(prompt_id)

            metrics_results = {}  # 分子累计
            total = 0             # 本 rank 样本数累计

            # 只在 rank0 展示进度条
            pbar = tqdm(test_loader, disable=not is_rank0(is_dist, rank))

            for step, batch in enumerate(pbar):
                inputs = batch[0]
                targets = batch[1]
                bs = len(targets)
                total += bs

                # inputs 是 dict，需要逐项搬到 GPU
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                # 可选：只在 rank0 的第一步打印一次看看输入
                if step == 0 and is_rank0(is_dist, rank):
                    try:
                        print(inputs)
                        print(targets[:5])
                    except Exception:
                        pass

                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                )

                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                topk_res = get_topk_results(
                    decoded,
                    scores,
                    targets,
                    args.num_beams,
                    all_items=all_items if getattr(args, "filter_items", False) else None
                )

                batch_metrics_res = get_metrics_results(topk_res, metrics)

                # 累加分子
                for m, res in batch_metrics_res.items():
                    metrics_results[m] = metrics_results.get(m, 0.0) + float(res)

                # rank0 展示一个“当前本 rank 的临时均值”（不是全局均值）
                if is_rank0(is_dist, rank):
                    temp = {m: (metrics_results[m] / total if total > 0 else 0.0) for m in metrics_results}
                    pbar.set_postfix(temp)

            # 分布式汇总成全局均值
            final_metrics = all_reduce_metrics(metrics_results, total, device, is_dist)

            all_prompt_results.append(final_metrics)

            if is_rank0(is_dist, rank):
                print("======================================================")
                print(f"Prompt {prompt_id} results: ", final_metrics)
                print("======================================================\n")

    # ===== 只在 rank0 计算 mean/min/max 并写文件 =====
    if is_rank0(is_dist, rank):
        mean_results, min_results, max_results = {}, {}, {}
        for m in metrics:
            all_res = [r.get(m, 0.0) for r in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res) if all_res else 0.0
            min_results[m] = min(all_res) if all_res else 0.0
            max_results[m] = max(all_res) if all_res else 0.0

        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")

        save_data = {
            "test_prompt_ids": prompt_ids,
            "mean_results": mean_results,
            "min_results": min_results,
            "max_results": max_results,
            "all_prompt_results": all_prompt_results,
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)

    # 清理分布式
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test (DDP supported)")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test(args)
