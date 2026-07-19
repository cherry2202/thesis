export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2

DATASET=Beauty
OUTPUT_DIR=./ckpt/$DATASET/aux8_2/

torchrun --nproc_per_node=3 --master_port=2320 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .nq_sid_tw32_rqvae.json \
    --coarse_index_file .nq_sid_tw8_rqvae.json \
    --resume_from_checkpoint ./ckpt/$DATASET/aux8_2/checkpoint-35277 \
    --temperature 1.0
