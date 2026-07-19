import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
from typing import List
# from transformers import EarlyStoppingCallback
from transformers import EarlyStoppingCallback, TrainerCallback

import torch
import transformers
import wandb  # 启用wandb

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from modeling_letter import LETTER
from utils import *
from collator import Collator

# ===================== 【只加这一句】 =====================
import torch.nn.functional as F
# ==========================================================




# ===================== 【只加这个函数】 =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits = torch.tensor(logits, device="cpu")
    # labels = torch.tensor(labels, device="cpu")
    with torch.no_grad():
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
    return {"eval_loss": loss.item()}
# ============================================================

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     with torch.no_grad():
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()

#         loss = F.cross_entropy(
#             shift_logits.view(-1, shift_logits.size(-1)),
#             shift_labels.view(-1),
#             ignore_index=-100
#         )
#     return {"eval_loss": loss.item()}

# 回调写在全局，不要塞进train()函数里
class LossLogCallback(TrainerCallback):
    def on_train_batch_end(self, args, state, control, outputs=None, **kwargs):
        if state.is_world_process_zero and outputs is not None:
            if hasattr(outputs, "fine_loss") and hasattr(outputs, "coarse_loss"):
                wandb.log({
                    "train/fine_loss": outputs.fine_loss.item(),
                    "train/coarse_loss": outputs.coarse_loss.item(),
                    "train/total_loss": outputs.loss.item(),
                }, step=state.global_step)


def train(args):
    print(torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    # 仅主进程初始化wandb，避免多进程重复创建实验
    if local_rank == 0:
        print(vars(args))
        wandb.init(
            project="TIGER-Rec",
            name=args.wandb_run_name,
            config=vars(args)
        )

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)


    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )
    args.deepspeed = None
    gradient_checkpointing= False


    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data[100])
        print(valid_data[100])


    collator = Collator(args, tokenizer)
    model = LETTER(config, args)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    # model.to(device)  # DDP自动分配设备，注释防止张量设备冲突
    if local_rank == 0:
        print(model)


    # if not ddp and torch.cuda.device_count() > 1:
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    # ===================== 核心修改：自定义前向计算 =====================
    def compute_loss(model, inputs, return_outputs=False):
        # 取出输入
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        coarse_labels = inputs.get("coarse_labels", None)

        # 前向传播（同时传入细+粗标签）
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            coarse_labels=coarse_labels  # 这是唯一新增！
        )

        return (outputs.loss, outputs) if return_outputs else outputs.loss

    # 替换 Trainer 的损失计算
    model.compute_loss = compute_loss
    # ==================================================================

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_step,
            optim=args.optim,
            eval_strategy=args.save_and_eval_strategy,  # 修复弃用参数警告
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            ddp_find_unused_parameters=False if ddp else None,
            prediction_loss_only=True,
            eval_accumulation_steps=2,
            report_to=["wandb"],  # 日志上报wandb
            run_name=args.wandb_run_name,
            logging_dir=os.path.join(args.output_dir, "tensorboard_logs"),
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 2000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=20),
            LossLogCallback()
        ]
    )
    model.config.use_cache = False


    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

    # 训练结束关闭wandb
    if local_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)