echo "_______________" >> logs/split_train_logs.out
sbatch -o logs/split_train_logs.out scripts/finetune/job_def.slurm