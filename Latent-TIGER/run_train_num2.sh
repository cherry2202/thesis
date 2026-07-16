export WANDB_MODE=online
export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=Beauty
NUM_LATENT=2
EXP_MODEL=$DATASET-latent-$NUM_LATENT
OUTPUT_DIR=./ablation/$EXP_MODEL/

torchrun --nproc_per_node=4 --master_port=4320 ./finetune.py \
    --resume_from_checkpoint ./ablation/Beauty-latent-2/checkpoint-768 \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --base_model t5-base \
    --num_latent_steps $NUM_LATENT \
    --per_device_batch_size 128 \
    --learning_rate 5e-4 \
    --epochs 12 \
    --save_and_eval_strategy steps \
    --index_file .tw32.json \
    --wandb_run_name $EXP_MODEL \
    --temperature 0.7 

