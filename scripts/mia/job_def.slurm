#!/bin/bash

#!/bin/bash
#SBATCH --job-name=loss_attacks
#SBATCH --time=2-00:00:00
#SBATCH --account=plgdyplomancipw-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu
#SBATCH --exclude=t0033

task_id=$1
DATASET=$2
ATTACK=$3
CHECKPOINT=$4
echo `pwd`
source ../.venv/bin/activate
bash scripts/mia/script.sh $task_id "nonmembers" $DATASET $ATTACK $CHECKPOINT
bash scripts/mia/script.sh $task_id "members" $DATASET $ATTACK $CHECKPOINT
echo "fin"