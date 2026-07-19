export HF_ENDPOINT=https://hf-mirror.com

DATASET=Beauty
DATA_PATH=../data
COARSE_DIM=8
ALPHA=1
EXP_MODEL=coarse${COARSE_DIM}_alpha${ALPHA}_1
CKPT_PATH=./ckpt/$DATASET/${EXP_MODEL}/
RESULTS_FILE=./results/${EXP_MODEL}/all-2ori.json

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
torchrun --nproc_per_node=4 --master_port=2300 test.py \
  --ckpt_path $CKPT_PATH \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --results_file $RESULTS_FILE \
  --test_batch_size 32 \
  --num_beams 20 \
  --test_prompt_ids 0 \
  --index_file .tw32.json