#!/bin/bash

# Fine tune XLM-Roberta on all data (train, validation, test).

export TRAIN_FILE=data/mlm_text/test-lm.txt
export VALID_FILE=data/mlm_text/validation-lm.txt
export OUTPUT_DIR=data/lang_model

python src/run_language_modeling.py \
  --model_type=xlm-roberta \
  --model_name_or_path=xlm-roberta-base \
  --do_train \
  --train_data_file=$TRAIN_FILE \
  --do_eval \
  --eval_data_file=$VALID_FILE \
  --output_dir=data/lang_model \
  --cache_dir=".hf_cache" \
  --mlm \
  --line_by_line \
  --num_train_epochs=2 \
  --learning_rate 7e-5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 5 \
  --save_total_limit 2 \
  --save_steps 2000 \
  --logging_dir=logs/mlm/train.log \
  --logging_first_step \
  --logging_steps=20 \
  --tensorboard_log_dir=tb_logs/mlm \
  --output_dir=data/lang_model \
  --overwrite_cache \
  --overwrite_output_dir \
  --seed 42
