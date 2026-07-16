DATASET=Beauty
DATA_PATH=../data
CKPT_PATH=./ablation/Beauty-latent-2
export HF_ENDPOINT=https://hf-mirror.com
NUM_LATENT=2

RESULTS_FILE=./ablation/$DATASET-latent-2.json
# SAVE_PRED_TXT=/home/jovyan/ceph-1/sujinsong/sujinsong/thesis/LETTER-master/LETTER-TIGER-new/predictions.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --master_port=2224 test_ddp.py \
  --ckpt_path $CKPT_PATH \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --results_file $RESULTS_FILE \
  --num_latent_steps $NUM_LATENT \
  --test_batch_size 16 \
  --num_beams 10 \
  --test_prompt_ids 0 \
  --index_file .tw32.json 
