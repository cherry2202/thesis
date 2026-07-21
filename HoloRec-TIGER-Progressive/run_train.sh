export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1

DATASET=Instruments
OUTPUT_DIR=./ckpt/$DATASET-progressive-prefix

torchrun --nproc_per_node=2 --master_port=4315 HoloRec-TIGER-Progressive/finetune.py \
  --output_dir $OUTPUT_DIR \
  --dataset $DATASET \
  --index_file .tw32.json \
  --per_device_batch_size 128 \
  --learning_rate 3e-4 \
  --epochs 200 \
  --prefix_levels 8,16,24 \
  --prefix_loss_weights 0.2,0.15,0.1 \
  --fine_loss_weight 1.0 \
  --prefix_latent_weight 0.5 \
  --temperature 0.8
