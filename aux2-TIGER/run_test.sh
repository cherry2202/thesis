export HF_ENDPOINT=https://hf-mirror.com
DATASET=Beauty
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/532-final.json
CKPT_PATH=./ckpt/$DATASET/aux_double_532/

# python test.py \
#     --gpu_id 2 \
#     --ckpt_path $CKPT_PATH \
#     --dataset $DATASET \
#     --data_path $DATA_PATH \
#     --results_file $RESULTS_FILE \
#     --test_batch_size 32 \
#     --num_beams 20 \
#     --test_prompt_ids 0 \
#     --index_file .index.xinyan32.epoch10000.alpha2e-2-beta1e-4.json


export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=2314 test.py \
  --ckpt_path $CKPT_PATH \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --results_file $RESULTS_FILE \
  --test_batch_size 32 \
  --num_beams 20 \
  --test_prompt_ids 0 \
  --index_file .nq_sid_tw32_rqvae.json