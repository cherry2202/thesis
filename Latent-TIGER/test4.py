# ========== 热补丁（可选，读取旧bin断点时必加，纯safetensors可注释掉） ==========
import sys
try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("transformers") and hasattr(mod, "check_torch_load_is_safe"):
            setattr(mod, "check_torch_load_is_safe", lambda *a, **k: None)
except Exception:
    pass
# ==========================================================================
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
    try:
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

        # ---------- 加载 tokenizer（增加local_files_only=True修复本地路径报错） ----------
        tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path, model_max_length=512, local_files_only=True)
        if local_rank == 0:
            print(f"Loaded tokenizer from checkpoint, vocab size: {len(tokenizer)}")

        # ---------- 加载 Latent 模型（增加local_files_only兼容safetensors） ----------
        model = LATENT.from_pretrained(
            args.ckpt_path,
            num_latent_steps=args.num_latent_steps,
            low_cpu_mem_usage=False,  # 多分布式关闭，避免权重冲突
            local_files_only=True,
        )
        # 推理必须开启KV缓存加速自回归生成
        model.config.use_cache = True
        # 将latent_steps存入config，方便后续读取
        model.config.num_latent_steps = args.num_latent_steps
        model.to(device)
        model.eval()
        torch.cuda.empty_cache()

        # ---------- 准备测试数据 ----------
        test_data = load_test_dataset(args)
        collator = TestCollator(args, tokenizer)
        all_items = test_data.get_all_items()

        # 候选词Trie过滤
        prefix_allowed_tokens = None
        if args.filter_items:
            candidate_trie = Trie(
                [[0] + tokenizer.encode(candidate) for candidate in all_items]
            )
            prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

        sampler = DistributedSampler(test_data, shuffle=False, drop_last=False)
        test_loader = DataLoader(
            test_data,
            batch_size=args.test_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

        if local_rank == 0:
            print("test data num:", len(test_data))
            print("Total batches per process:", len(test_loader))

        # ---------- 评估循环 ----------
        metrics = args.metrics.split(",")
        # 从命令行参数读取多prompt，不再硬编码[0]
        prompt_ids = [int(pid.strip()) for pid in args.test_prompt_ids.split(",")]
        if local_rank == 0:
            print(f"Running test prompt ids: {prompt_ids}")

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

                    # 推理关闭梯度计算，节省显存加速
                    with torch.no_grad():
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
                            num_latent_steps=args.num_latent_steps, # 关键：透传隐思考步数
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
                        temp = {m: metrics_results[m] / total, 4) for m in metrics}
                        print(f"Step {step}, interim metric: {temp}")

                    # 每10步打印前3条样本的真实标签 + top5 beam预测
                    # if step % 10 == 0:
                    #     print("=" * 60)
                    #     print(f"【Step {step} 样本预测示例】")
                    #     batch_show_num = min(3, len(targets))
                    #     for i in range(batch_show_num):
                    #         true_item = targets[i]
                    #         # 取出该样本前5个beam结果
                    #         top5_pred = output_texts[i * args.num_beams : i * args.num_beams + 5]
                    #         print(f"样本{i} 真实Item: {true_item}")
                    #         print(f"Top5预测: {top5_pred}")
                    #     print("=" * 60 + "\n")

            except Exception as e:
                print(f"Rank {local_rank} error when prompt {prompt_id}: {e}")
                traceback.print_exc()
                raise

            # ---------- 多进程指标聚合 ----------
            metric_tensors = {
                m: torch.tensor(metrics_results[m], device=device, dtype=torch.float64)
                for m in metrics
            }
            total_tensor = torch.tensor(total, device=device, dtype=torch.float64)

            dist.barrier()
            for m in metric_tensors:
                dist.all_reduce(metric_tensors[m], op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            dist.barrier()

            if local_rank == 0:
                final_metrics = {
                    m: round(metric_tensors[m].item() / total_tensor.item(), 4)
                    for m in metrics
                }
                print("=" * 60)
                print(f"Prompt {prompt_id} Final Metrics: {final_metrics}")
                print("=" * 60)
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
                "num_latent_steps": args.num_latent_steps,
                "beam_size": args.num_beams,
                "final_results": final_output
            }
            with open(args.results_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            print(f"All results saved to {args.results_file}")

    finally:
        # 无论正常退出还是报错，都销毁分布式进程组，防止卡死
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec distributed test with Latent reasoning")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    # Latent推理专属参数
    parser.add_argument("--num_latent_steps", type=int, default=2,
                        help="Number of latent reasoning steps before token generation")
    args = parser.parse_args()
    test(args)