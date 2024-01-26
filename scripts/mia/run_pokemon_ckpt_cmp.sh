export LOGS=$(echo "logs/mia`date`.out" | tr -d '[:blank:]')
for ATTACK in "embeddings";
do
    for DATASET in "pokemons";
    do
        for CHECKPOINT in -1 5000 10000 15000 20000 25000 30000;
        do
            for i in {0..15};
            do
                sbatch -o $LOGS scripts/mia/job_def.slurm --export task_id=$i,DATASET=$DATASET,ATTACK=$ATTACK,CHECKPOINT=$CHECKPOINT
            done
        done
    done
done