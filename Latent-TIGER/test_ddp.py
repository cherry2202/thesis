import argparse
import json
import os
import sys
import traceback
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config
from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie
from modeling_latent import LATENT

def test(args):
    # ---------- 分布式初始化 ----------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://")
    
    # 每个进程使用不同的随机种子
    set_seed(args.seed + local_rank)

    if local_rank == 0:
        print(vars(args))

    # ---------- 加载 tokenizer 和扩充词表 ----------
    tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)
    train_data, _ = load_datasets(args)   # 需要确保 load_datasets 能正确返回
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("train data num:", len(train_data))

    # ---------- 加载模型 ----------
    model = LATENT.from_pretrained(
        args.ckpt_path,
        num_latent_steps=args.num_latent_steps,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    # ---------- 准备测试数据 ----------
    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    if args.filter_items:
        candidate_trie = Trie(
            [[0] + tokenizer.encode(candidate) for candidate in all_items]
        )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
    else:
        prefix_allowed_tokens = None

    sampler = DistributedSampler(test_data, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    if local_rank == 0:
        print("test data num:", len(test_data))
        print("Total batches per process:", len(test_loader))

    # ---------- 评估循环 ----------
    metrics = args.metrics.split(",")
    prompt_ids = [0]  # 可扩展

    all_prompt_results = []

    for prompt_id in prompt_ids:
        test_loader.dataset.set_prompt(prompt_id)
        sampler.set_epoch(0)

        metrics_results = {m: 0.0 for m in metrics}
        total = 0

        iterator = tqdm(test_loader, desc=f"Prompt {prompt_id}", disable=local_rank != 0)

        try:
            for step, batch in enumerate(iterator):
                inputs = {k: v.to(device) for k, v in batch[0].items()}
                targets = batch[1]
                batch_size = len(targets)
                total += batch_size

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
                output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                topk_res = get_topk_results(
                    output_texts, scores, targets, args.num_beams,
                    all_items=all_items if args.filter_items else None
                )
                batch_metrics_res = get_metrics_results(topk_res, metrics)

                for m, val in batch_metrics_res.items():
                    metrics_results[m] += val

                if local_rank == 0 and step % 50 == 0:
                    temp = {m: metrics_results[m] / total for m in metrics}
                    print(f"Step {step}, interim: {temp}")

        except Exception as e:
            print(f"Rank {local_rank} encountered error: {e}")
            traceback.print_exc()
            # 将错误信息通过 all_reduce 传播？但这里我们简单将 metrics 置零并继续
            # 但最好直接 raise，因为一个进程失败会导致通信失败
            raise

        # ---------- 跨进程聚合 ----------
        metric_tensors = {m: torch.tensor(metrics_results[m], device=device, dtype=torch.float64) for m in metrics}
        total_tensor = torch.tensor(total, device=device, dtype=torch.float64)

        # 同步所有进程，确保都到达聚合点
        dist.barrier()

        for m in metric_tensors:
            dist.all_reduce(metric_tensors[m], op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        # 再次同步，确保聚合完成
        dist.barrier()

        if local_rank == 0:
            final_metrics = {m: (metric_tensors[m].item() / total_tensor.item()) for m in metrics}
            print("======================================================")
            print(f"Prompt {prompt_id} results: {final_metrics}")
            print("======================================================")
            all_prompt_results.append(final_metrics)

    # ---------- 主进程保存结果 ----------
    if local_rank == 0:
        if len(all_prompt_results) == 1:
            final_output = all_prompt_results[0]
        else:
            mean_results = {}
            min_results = {}
            max_results = {}
            for m in metrics:
                all_res = [res[m] for res in all_prompt_results]
                mean_results[m] = sum(all_res) / len(all_res)
                min_results[m] = min(all_res)
                max_results[m] = max(all_res)
            final_output = {
                "mean": mean_results,
                "min": min_results,
                "max": max_results,
                "all_prompt_results": all_prompt_results
            }

        save_data = {
            "test_prompt_ids": args.test_prompt_ids,
            "final_results": final_output
        }
        with open(args.results_file, "w") as f:
            json.dump(save_data, f, indent=4)
        print(f"Results saved to {args.results_file}")

    # ---------- 清理 ----------
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test_distributed")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser.add_argument("--num_latent_steps", type=int, default=2,
                        help="Number of latent reasoning steps before generation")
    args = parser.parse_args()
    test(args)