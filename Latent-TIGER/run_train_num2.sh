export OMP_NUM_THREADS=4
export WANDB_MODE=online
export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_LOAD_ALLOW_UNSAFE=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 实验配置
DATASET=Beauty
NUM_LATENT=2
EXP_MODEL=$DATASET-latent-$NUM_LATENT
# 区分续训输出目录，避免覆盖原有checkpoint
OUTPUT_DIR=./ablation/${EXP_MODEL}_new/
# 断点路径
# RESUME_CKPT=./ablation/Beauty-latent-2_resume/checkpoint-1400
# 从头训练（已移除断点续训注释行，无多余\语法污染）
torchrun --nproc_per_node=4 --master_port=4320 ./finetune.py \
    --output_dir ${OUTPUT_DIR} \
    --dataset ${DATASET} \
    --base_model t5-base \
    --num_latent_steps ${NUM_LATENT} \
    --per_device_batch_size 128 \
    --learning_rate 2e-4 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.1 \
    --epochs 400 \
    --save_and_eval_strategy steps \
    --save_and_eval_steps 200 \
    --index_file .tw32.json \
    --wandb_run_name ${EXP_MODEL}_new \
    --temperature 0.7 \
    --seed 42 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --fp16