export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Beauty
OUTPUT_DIR=./ckpt/$DATASET/aux16_2/

torchrun --nproc_per_node=4 --master_port=2314 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 100 \
    --index_file .nq_sid_tw32_rqvae.json \
    --coarse_index_file .nq_sid_tw16_rqvae.json \
    --temperature 1.0
