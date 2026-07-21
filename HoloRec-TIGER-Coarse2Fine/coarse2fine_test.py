import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import T5Config, T5Tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "HoloRec-TIGER"))
sys.path.insert(0, BASE_DIR)

from coarse_index import CoarseFineIndex
from collator import TestCollator
from evaluate import get_metrics_results, get_topk_results
from generation_trie import Trie, build_trie_from_token_sequences
from modeling_letter import LETTER
from test import decoder_prefill, decoder_step, index_select_past_key_values
from utils import load_datasets, load_test_dataset, parse_dataset_args, parse_global_args, parse_test_args, set_seed


def init_distributed():
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


def ensure_parent_dir(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_code_text(x: str) -> str:
    return str(x).strip().replace(" ", "")


def fine_codes_to_text(fine_codes: List[str]) -> str:
    return normalize_code_text("".join(fine_codes))


def infer_fine_code_length(test_data) -> int:
    first_key = next(iter(test_data.indices.keys()))
    return len(test_data.indices[first_key])


def build_fine_trie_from_index(coarse_fine_index: CoarseFineIndex) -> Trie:
    return Trie(coarse_fine_index.fine_sequences)


def build_allowed_cache_for_beams(fine_prefixes: List[List[int]], fine_trie: Trie, cache: Dict[tuple, List[int]]):
    allowed_list = []
    for prefix in fine_prefixes:
        key = tuple(prefix)
        if key not in cache:
            cache[key] = fine_trie.get(prefix)
        allowed_list.append(cache[key])
    return allowed_list


def pack_past_key_values(beams):
    packed_past = []
    num_layers = len(beams[0]["past_key_values"])
    for layer_idx in range(num_layers):
        layer_parts = []
        part_num = len(beams[0]["past_key_values"][layer_idx])
        for part_idx in range(part_num):
            layer_parts.append(torch.cat([b["past_key_values"][layer_idx][part_idx] for b in beams], dim=0))
        packed_past.append(tuple(layer_parts))
    return tuple(packed_past)


def coarse_filter_adjustment(
    coarse_log_probs: torch.Tensor,
    compatible_coarse: List[int],
    coarse_top_ids: torch.Tensor,
    mode: str,
    guidance_weight: float,
    mismatch_penalty: float,
    device: torch.device,
):
    if not compatible_coarse:
        return None if mode == "hard" else -mismatch_penalty

    compatible_tensor = torch.tensor(compatible_coarse, dtype=torch.long, device=device)
    coarse_score = float(coarse_log_probs.index_select(0, compatible_tensor).max().item())
    top_set = set(int(x) for x in coarse_top_ids.tolist())
    matched = any(int(x) in top_set for x in compatible_coarse)

    if mode == "hard" and not matched:
        return None
    if mode == "hybrid" and not matched:
        return guidance_weight * coarse_score - mismatch_penalty
    return guidance_weight * coarse_score


@torch.no_grad()
def batch_generate_coarse_filtered(
    model,
    tokenizer,
    inputs,
    fine_trie: Trie,
    coarse_fine_index: CoarseFineIndex,
    fine_code_len: int,
    num_beams: int,
    coarse_topk_per_step: int,
    coarse_filter_mode: str,
    coarse_guidance_weight: float,
    coarse_mismatch_penalty: float,
):
    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size = input_ids.size(0)

    encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    enc_h0 = encoder_outputs.last_hidden_state
    enc_m0 = attention_mask

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    start_ids = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)

    init_last_hidden, init_past = decoder_prefill(
        model=model,
        encoder_hidden_states=enc_h0,
        encoder_attention_mask=enc_m0,
        decoder_input_ids=start_ids,
    )

    beams_by_sample = []
    for b in range(batch_size):
        beams_by_sample.append([
            {
                "fine_token_ids": [],
                "score": 0.0,
                "last_hidden": init_last_hidden[b:b + 1].contiguous(),
                "past_key_values": index_select_past_key_values(
                    init_past, torch.tensor([b], dtype=torch.long, device=device)
                ),
            }
        ])

    trie_cache = {}

    for step_idx in range(fine_code_len):
        flat_beams = []
        flat_sample_ids = []
        for sample_id in range(batch_size):
            for beam in beams_by_sample[sample_id]:
                flat_beams.append(beam)
                flat_sample_ids.append(sample_id)
        if not flat_beams:
            break

        flat_sample_ids_t = torch.tensor(flat_sample_ids, dtype=torch.long, device=device)
        flat_last_hidden = torch.cat([b["last_hidden"] for b in flat_beams], dim=0)
        packed_past = pack_past_key_values(flat_beams)
        flat_enc_h = enc_h0.index_select(0, flat_sample_ids_t)
        flat_enc_m = enc_m0.index_select(0, flat_sample_ids_t)

        coarse_logits = model.coarse_head(flat_last_hidden)
        temp = getattr(model, "temperature", 1.0)
        coarse_probs = F.softmax(coarse_logits / temp, dim=-1)
        coarse_emb = torch.matmul(coarse_probs, model.shared.weight)
        coarse_log_probs = F.log_softmax(coarse_logits, dim=-1)
        topk = min(coarse_topk_per_step, coarse_log_probs.size(-1))
        coarse_top_ids = torch.topk(coarse_log_probs, k=topk, dim=-1).indices

        hidden_after_coarse, past_after_coarse = decoder_step(
            model=model,
            encoder_hidden_states=flat_enc_h,
            encoder_attention_mask=flat_enc_m,
            past_key_values=packed_past,
            input_ids=None,
            inputs_embeds=coarse_emb.unsqueeze(1),
        )

        fine_hidden = model._inject_coarse(hidden_after_coarse, coarse_emb)
        fine_logits = model.fine_head(fine_hidden)
        fine_log_probs = F.log_softmax(fine_logits, dim=-1)

        fine_prefixes = [b["fine_token_ids"] for b in flat_beams]
        allowed_list = build_allowed_cache_for_beams(fine_prefixes, fine_trie, trie_cache)

        candidate_parent_flat_idx = []
        candidate_sample_ids = []
        candidate_token_ids = []
        candidate_scores = []
        candidate_fine_prefixes = []

        for flat_idx, beam in enumerate(flat_beams):
            allowed = allowed_list[flat_idx]
            if not allowed:
                continue
            allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
            allowed_scores = fine_log_probs[flat_idx].index_select(0, allowed_tensor)
            expand_k = min(max(1, num_beams * 2), allowed_tensor.size(0))
            fine_top_scores, fine_top_pos = torch.topk(allowed_scores, k=expand_k)
            fine_top_ids = allowed_tensor[fine_top_pos]

            for j in range(expand_k):
                fine_token_id = int(fine_top_ids[j].item())
                new_prefix = beam["fine_token_ids"] + [fine_token_id]
                compatible = coarse_fine_index.compatible_coarse_ids(step_idx, new_prefix)
                coarse_adjustment = coarse_filter_adjustment(
                    coarse_log_probs=coarse_log_probs[flat_idx],
                    compatible_coarse=compatible,
                    coarse_top_ids=coarse_top_ids[flat_idx],
                    mode=coarse_filter_mode,
                    guidance_weight=coarse_guidance_weight,
                    mismatch_penalty=coarse_mismatch_penalty,
                    device=device,
                )
                if coarse_adjustment is None:
                    continue

                candidate_parent_flat_idx.append(flat_idx)
                candidate_sample_ids.append(flat_sample_ids[flat_idx])
                candidate_token_ids.append(fine_token_id)
                candidate_scores.append(float(beam["score"]) + float(fine_top_scores[j].item()) + coarse_adjustment)
                candidate_fine_prefixes.append(new_prefix)

        if not candidate_token_ids:
            break

        candidate_parent_flat_idx_t = torch.tensor(candidate_parent_flat_idx, dtype=torch.long, device=device)
        candidate_input_ids = torch.tensor(candidate_token_ids, dtype=torch.long, device=device).unsqueeze(1)
        cand_enc_h = flat_enc_h.index_select(0, candidate_parent_flat_idx_t)
        cand_enc_m = flat_enc_m.index_select(0, candidate_parent_flat_idx_t)
        cand_past = index_select_past_key_values(past_after_coarse, candidate_parent_flat_idx_t)

        cand_hidden_after_fine, cand_past_after_fine = decoder_step(
            model=model,
            encoder_hidden_states=cand_enc_h,
            encoder_attention_mask=cand_enc_m,
            past_key_values=cand_past,
            input_ids=candidate_input_ids,
            inputs_embeds=None,
        )

        candidates_grouped = [[] for _ in range(batch_size)]
        for idx, sample_id in enumerate(candidate_sample_ids):
            candidates_grouped[sample_id].append(idx)

        new_beams_by_sample = [[] for _ in range(batch_size)]
        for sample_id in range(batch_size):
            cand_indices = candidates_grouped[sample_id]
            cand_indices.sort(key=lambda i: candidate_scores[i], reverse=True)
            seen = set()
            for i in cand_indices:
                key = tuple(candidate_fine_prefixes[i])
                if key in seen:
                    continue
                seen.add(key)
                new_beams_by_sample[sample_id].append(
                    {
                        "fine_token_ids": candidate_fine_prefixes[i],
                        "score": candidate_scores[i],
                        "last_hidden": cand_hidden_after_fine[i:i + 1].contiguous(),
                        "past_key_values": index_select_past_key_values(
                            cand_past_after_fine, torch.tensor([i], dtype=torch.long, device=device)
                        ),
                    }
                )
                if len(new_beams_by_sample[sample_id]) >= num_beams:
                    break
        beams_by_sample = new_beams_by_sample

    decoded = []
    scores = []
    for sample_id in range(batch_size):
        beams = beams_by_sample[sample_id] or [{"fine_token_ids": [], "score": -1e9}]
        while len(beams) < num_beams:
            beams.append({"fine_token_ids": list(beams[-1]["fine_token_ids"]), "score": float(beams[-1]["score"])})
        for beam in beams[:num_beams]:
            decoded.append(tokenizer.decode(beam["fine_token_ids"], skip_special_tokens=True).replace(" ", ""))
            scores.append(float(beam["score"]))
    return decoded, scores


def all_reduce_metrics(metrics_results: dict, total: int, device: torch.device, is_dist: bool):
    if not is_dist:
        return {k: (float(v) / float(total) if total > 0 else 0.0) for k, v in metrics_results.items()}
    total_tensor = torch.tensor([total], device=device, dtype=torch.long)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    total_all = int(total_tensor.item())
    out = {}
    for key, value in metrics_results.items():
        tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        out[key] = tensor.item() / total_all if total_all > 0 else 0.0
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Coarse-filtered HoloRec-TIGER inference")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)
    parser.add_argument("--coarse_index_file", type=str, default=".tw8.json")
    parser.add_argument("--save_pred_txt", type=str, default="./results/coarse2fine_predictions.txt")
    parser.add_argument("--coarse_filter_mode", choices=["soft", "hard", "hybrid"], default="hybrid")
    parser.add_argument("--coarse_topk_per_step", type=int, default=16)
    parser.add_argument("--coarse_guidance_weight", type=float, default=0.2)
    parser.add_argument("--coarse_mismatch_penalty", type=float, default=1.0)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    return parser.parse_args()


def test(args):
    is_dist, local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)
    set_seed(getattr(args, "seed", 42) + rank)

    config = T5Config.from_pretrained(args.ckpt_path, local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained(args.ckpt_path, model_max_length=512, local_files_only=True)
    train_data, _ = load_datasets(args)
    tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    model = LETTER.from_pretrained(args.ckpt_path, config=config, local_files_only=True)
    model.resize_token_embeddings_and_heads(len(tokenizer))
    model.to(device)
    model.eval()

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer)
    coarse_fine_index = CoarseFineIndex(test_data, tokenizer)
    fine_trie = build_fine_trie_from_index(coarse_fine_index)
    fine_code_len = infer_fine_code_length(test_data)
    all_items = [normalize_code_text(x) for x in coarse_fine_index.all_fine_texts]

    sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False) if is_dist else None
    loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)
    metrics = args.metrics.split(",")
    metrics_results = {}
    total = 0
    local_pred_lines = []

    amp_dtype = torch.bfloat16 if args.use_bf16 else torch.float16 if args.use_fp16 else None
    pbar = tqdm(loader, disable=not is_rank0(is_dist, rank))
    with torch.no_grad():
        for batch in pbar:
            inputs, raw_targets = batch
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            target_texts = [fine_codes_to_text(t["fine_codes"]) for t in raw_targets]
            total += len(target_texts)

            if amp_dtype is not None:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    decoded, scores = batch_generate_coarse_filtered(model, tokenizer, inputs, fine_trie, coarse_fine_index, fine_code_len, args.num_beams, args.coarse_topk_per_step, args.coarse_filter_mode, args.coarse_guidance_weight, args.coarse_mismatch_penalty)
            else:
                decoded, scores = batch_generate_coarse_filtered(model, tokenizer, inputs, fine_trie, coarse_fine_index, fine_code_len, args.num_beams, args.coarse_topk_per_step, args.coarse_filter_mode, args.coarse_guidance_weight, args.coarse_mismatch_penalty)

            for i, tgt in enumerate(target_texts):
                local_pred_lines.append(f"target: {tgt}")
                for beam_idx, pred in enumerate(decoded[i * args.num_beams:(i + 1) * args.num_beams], start=1):
                    local_pred_lines.append(f"pred_{beam_idx}: {normalize_code_text(pred)}")
                local_pred_lines.append("")

            topk_res = get_topk_results(decoded, scores, target_texts, args.num_beams, all_items=all_items if args.filter_items else None)
            batch_metrics = get_metrics_results(topk_res, metrics)
            for key, value in batch_metrics.items():
                metrics_results[key] = metrics_results.get(key, 0.0) + float(value)
            if is_rank0(is_dist, rank):
                pbar.set_postfix({k: metrics_results[k] / total for k in metrics_results})

    final_metrics = all_reduce_metrics(metrics_results, total, device, is_dist)
    if is_rank0(is_dist, rank):
        ensure_parent_dir(args.results_file)
        with open(args.results_file, "w", encoding="utf-8") as f:
            json.dump({"results": final_metrics, "args": vars(args)}, f, indent=4, ensure_ascii=False)
        ensure_parent_dir(args.save_pred_txt)
        with open(args.save_pred_txt, "w", encoding="utf-8") as f:
            for line in local_pred_lines:
                f.write(line + "\n")
        print(final_metrics)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    test(parse_args())
