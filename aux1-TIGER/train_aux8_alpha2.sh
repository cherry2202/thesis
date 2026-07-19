export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Beauty
OUTPUT_DIR=./ckpt/$DATASET/aux8_alpha2/

torchrun --nproc_per_node=4 --master_port=2214 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 512 \
    --learning_rate 2e-3 \
    --epochs 200 \
    --index_file .nq_sid_tw32_rqvae.json \
    --coarse_index_file .nq_sid_tw8_rqvae.json \
    --alpha 0.2 \
    --num_workers 16 \
    --temperature 1.0
