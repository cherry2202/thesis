import argparse
import os
import sys
from typing import List
from transformers import EarlyStoppingCallback

import torch
import transformers
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from modeling_latent import LATENT
from utils import *
from collator import Collator

def train(args):
    print("CUDA available:", torch.cuda.is_available())

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if local_rank == 0:
        print("==== Training Arguments ====")
        print(vars(args))

    device = torch.device("cuda", local_rank)
    gradient_checkpointing = False

    # 加载底座配置&分词器，local_files_only=True修复本地路径HF报错
    config = T5Config.from_pretrained(args.base_model, local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
        local_files_only=True,
    )

    # 加载数据集，扩充词表
    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"Added {add_num} new tokens, total vocab size: {len(tokenizer)}")
        print(f"Train dataset total samples: {len(train_data)}")
        # 保存分词器与配置到输出目录
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print("Sample train data:", train_data[100])
        print("Sample valid data:", valid_data[100])

    collator = Collator(args, tokenizer)

    # 初始化自定义Latent模型
    model = LATENT(config, num_latent_steps=args.num_latent_steps)
    # 同步扩充词表权重（现已简化为父类原生resize）
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if local_rank == 0:
        print("==== Model Structure ====")
        print(model)

    # 训练参数完整配置
    training_args = transformers.TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_step,
        optim=args.optim,

        # 保存&评估策略
        eval_strategy=args.save_and_eval_strategy,
        save_strategy=args.save_and_eval_strategy,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        output_dir=args.output_dir,
        save_total_limit=2,
        load_best_model_at_end=True,
        eval_delay=1 if args.save_and_eval_strategy == "epoch" else 2000,

        # 多卡DDP核心修复：全局关闭未使用参数检测
        ddp_find_unused_parameters=False,

        # 梯度与显存优化
        max_grad_norm=3.0,
        fp16=True,
        gradient_checkpointing=gradient_checkpointing,

        # 日志
        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
    )

    # 训练阶段强制关闭KV缓存，统一latent推理无缓存逻辑
    model.config.use_cache = False

    # 断点续训/从头训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 保存训练状态与最优模型
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)
    if local_rank == 0:
        print(f"Training finished, best model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Latent-T5 LLMRec Training Script')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    # Latent隐推理步数参数
    parser.add_argument("--num_latent_steps", type=int, default=2,
                        help="Number of latent reasoning steps before target decoding")

    args = parser.parse_args()
    train(args)