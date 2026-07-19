# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com

DATASET=Instruments
COARSE_DIM=8
ALPHA=1
EXP_MODEL=${DATASET}_coarse${COARSE_DIM}_alpha${ALPHA}_1
OUTPUT_DIR=./ckpt/$DATASET/${EXP_MODEL}/


torchrun --nproc_per_node=4 --master_port=2320 ./finetune2.py \
    --base_model t5-base \
    --output_dir $OUTPUT_DIR \
    --wandb_run_name ${EXP_MODEL} \
    --dataset $DATASET \
    --per_device_batch_size 128 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .tw32.json \
    --coarse_index_file .tw${COARSE_DIM}.json \
    --alpha ${ALPHA} \
    --temperature 1.0
