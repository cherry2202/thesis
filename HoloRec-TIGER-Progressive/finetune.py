import argparse
import os
import random

import numpy as np
import torch
import transformers
from torch.utils.data import ConcatDataset
from transformers import EarlyStoppingCallback, T5Config, T5Tokenizer

from collator import ProgressiveCollator
from data import SeqRecDataset
from modeling_progressive import ProgressiveLETTER


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Progressive-prefix HoloRec-TIGER")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default="./ckpt/TIGER")
    parser.add_argument("--output_dir", type=str, default="./ckpt/progressive-tiger")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="Instruments")
    parser.add_argument("--index_file", type=str, default=".tw32.json")
    parser.add_argument("--max_his_len", type=int, default=20)
    parser.add_argument("--add_prefix", action="store_true", default=False)
    parser.add_argument("--sample_num", type=int, default=-1)

    parser.add_argument("--prefix_levels", type=str, default="8,16,24")
    parser.add_argument("--prefix_loss_weights", type=str, default="0.2,0.15,0.1")
    parser.add_argument("--fine_loss_weight", type=float, default=1.0)
    parser.add_argument("--prefix_latent_weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="progressive-tiger")
    parser.add_argument("--report_to", type=str, default="none")
    return parser.parse_args()


def _parse_csv_numbers(text, cast=float):
    return [cast(x) for x in text.split(",") if x]


def train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if ddp:
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)

    train_data = ConcatDataset([SeqRecDataset(args, mode="train", sample_num=args.sample_num)])
    valid_data = SeqRecDataset(args, mode="valid")
    tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())

    prefix_levels = _parse_csv_numbers(args.prefix_levels, int)
    prefix_loss_weights = _parse_csv_numbers(args.prefix_loss_weights, float)

    model = ProgressiveLETTER.from_pretrained(args.base_model, config=config)
    model.configure_progressive(
        prefix_levels=prefix_levels,
        prefix_loss_weights=prefix_loss_weights,
        fine_loss_weight=args.fine_loss_weight,
        prefix_latent_weight=args.prefix_latent_weight,
        temperature=args.temperature,
    )
    model.resize_token_embeddings_and_heads(len(tokenizer))
    model.to(device)

    tokenizer.save_pretrained(args.output_dir)
    model.config.save_pretrained(args.output_dir)

    if local_rank == 0:
        print(vars(args))
        print("train data num:", len(train_data))
        print("valid data num:", len(valid_data))
        print("prefix levels:", prefix_levels)
        print("prefix loss weights:", prefix_loss_weights)
        print(train_data[0])

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
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
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            remove_unused_columns=False,
            report_to=args.report_to,
            run_name=args.wandb_run_name,
        ),
        processing_class=tokenizer,
        data_collator=ProgressiveCollator(args, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    train(parse_args())
