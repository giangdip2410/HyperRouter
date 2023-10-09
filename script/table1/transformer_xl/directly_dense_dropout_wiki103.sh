#!/bin/bash
echo 'Run training...'
python -u train.py \
    --cuda \
    --data ../data/wikitext-103/ \
    --dataset wt103 \
    --n_layer 4 \
    --d_model 256 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 8192 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 0 \
    --max_step 134000 \
    --tgt_len 512 \
    --mem_len 512 \
    --eval_tgt_len 128 \
    --batch_size 66 \
    --multi_gpu \
    --work_dir Directly_Dense_Dropout_Training
