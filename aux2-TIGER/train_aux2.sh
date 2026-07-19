export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Beauty
OUTPUT_DIR=./ckpt/$DATASET/aux_double/

torchrun --nproc_per_node=4 --master_port=2222 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 64 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .nq_sid_tw32_rqvae.json \
    --coarse_index_file .nq_sid_tw8_rqvae.json \
    --mid_index_file .nq_sid_tw16_rqvae.json \
    --alpha_fine 0.8 \
    --alpha_mid 0.1 \
    --alpha_coarse 0.1 \
    --temperature 1.0
