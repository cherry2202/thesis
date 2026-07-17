export OMP_NUM_THREADS=4
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export SAFETENSORS_FAST_GPU=1
export TRANSFORMERS_SAFETENSORS_ONLY=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET=Beauty
DATA_PATH=../data
CKPT_PATH=./ablation/Beauty-latent-2_new/checkpoint-5800
NUM_LATENT=2
RESULTS_FILE=./ablation/rst-{$DATASET}-latent-2-new.json

torchrun --nproc_per_node=4 --master_port=2224 test4.py \
  --ckpt_path $CKPT_PATH \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --results_file $RESULTS_FILE \
  --num_latent_steps $NUM_LATENT \
  --test_batch_size 16 \
  --num_beams 10 \
  --test_prompt_ids 0 \
  --filter_items \
  --index_file .tw32.json