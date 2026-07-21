DATASET=Instruments
DATA_PATH=../data
CKPT_PATH=./ckpt/$DATASET-ceshi/checkpoint-16254
RESULTS_FILE=./results/$DATASET/coarse2fine-hybrid.json
SAVE_PRED_TXT=./results/$DATASET/coarse2fine_predictions.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --master_port=2216 HoloRec-TIGER-Coarse2Fine/coarse2fine_test.py \
  --ckpt_path $CKPT_PATH \
  --dataset $DATASET \
  --data_path $DATA_PATH \
  --results_file $RESULTS_FILE \
  --save_pred_txt $SAVE_PRED_TXT \
  --test_batch_size 32 \
  --num_beams 20 \
  --index_file .tw32.json \
  --coarse_index_file .tw8.json \
  --coarse_filter_mode hybrid \
  --coarse_topk_per_step 16 \
  --coarse_guidance_weight 0.2 \
  --coarse_mismatch_penalty 1.0
