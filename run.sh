#!/bin/bash

python main.py \
  --task asqp \
  --dataset rest15 \
  --model_name_or_path t5-base \
  --n_gpu 0 \
  --do_train \
  --do_direct_eval \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 32 \
  --learning_rate 3e-4 \
  --max_seq_length 256 \
  --num_train_epochs 20 \
  --seed 666

python main.py \
  --task asqp \
  --dataset rest15 \
  --model_name_or_path t5-base \
  --n_gpu 0 \
  --do_correction \
  --correction_data_path ./data/rest15/Correction.txt \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 16 \
  --use_first_stage_weights \
  --learning_rate 1e-4 \
  --max_seq_length 256 \
  --num_correction_epochs 25 \
  --weight_decay 0 \
  --warmup_steps 80 \
  --seed 666

python main.py \
  --task asqp \
  --dataset rest15 \
  --model_name_or_path t5-base \
  --n_gpu 0 \
  --do_direct_eval \
  --eval_with_correction \
  --eval_batch_size 32 \
  --max_seq_length 256 \
  --correction_checkpoint "correction_checkpoint_epoch_25"
