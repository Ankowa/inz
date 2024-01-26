#!/bin/bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export train_data_dir="data/pokemon-split-200-train"
export LOGGING_DIR="logs/tensorboard-logs"

source .venv/bin/activate
accelerate launch --mixed_precision="fp16" pokemon_finetuning/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$train_data_dir \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=30000 \
  --checkpointing_steps=5000 \
  --learning_rate=4e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="models/pokemon-split-200-sd" \
  --logging_dir=$LOGGING_DIR
