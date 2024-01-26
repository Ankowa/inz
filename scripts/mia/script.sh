#!/bin/bash

export TASK_ID=$1
export OUTPUT=$2
export DATASET=$3
export ATTACK=$4
export CHECKPOINT_ID=$5

if [ "$DATASET" = "laion_mi" ]; then
    export MODEL_NAME="CompVis/stable-diffusion-v1-4"
    export OUTPUT_DIR="LAION"
    export NUM_IMAGES=5100
elif [ "$DATASET" = "pokemons" ]; then
    export INPUT_DIR="pokemon-split"
    export MODEL_NAME="sd-pokemon-model-full-run-with-split"
    export OUTPUT_DIR="pokemons"
    export NUM_IMAGES=200
else
    echo "Wrong dataset name"
    exit 1
fi

export OUTPUT_FILENAME=${OUTPUT}-${ATTACK}.npz

python3 -u attack/mia.py \
  --input_dir=$INPUT_DIR \
  --model_name=$MODEL_NAME \
  --model_dir=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR \
  --output_filename=$OUTPUT_FILENAME \
  --num_images=$NUM_IMAGES \
  --batch_size=$BATCH_SIZE \
  --parallel_tasks_cnt=16 \
  --task_id=$TASK_ID \
  --checkpoint_id=$CHECKPOINT_ID \
  --attack=$ATTACK