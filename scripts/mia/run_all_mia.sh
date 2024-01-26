export LOGS=$(echo "logs/mia`date`.out" | tr -d '[:blank:]')
for ATTACK in "embeddings" "embeddings-diff-starts" "embeddings-shift" "embeddings-text-shift" "embeddings-text-shift-05" "embeddings-wrong-start" "embeddings-wrong-start-longer" "embeddings-wrong-start-longer-text" "embeddings-wrong-start-longer-50" "embeddings-wrong-start-longer-300" "embeddings-wrong-start-300-from-200" "embeddings-reversed-noise" "embeddings-no-classifier-free-guidance" "embeddings-noise-100-everytime" "embeddings-wrong-start-reversed-steps" "embeddings-reversed-noise-big-classifier-free-guidance" "black-box" "embeddings-carlini";
do
    for DATASET in "laion_mi" "pokemons";
    do
        for i in {0..15};
        do
            sbatch -o $LOGS scripts/mia/job_def.slurm --export task_id=$i,DATASET=$DATASET,ATTACK=$ATTACK,CHECKPOINT=-1
        done
    done
done