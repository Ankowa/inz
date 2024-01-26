python3 pokemon_finetuning/split_dataset.py \
    --dataset_name="lambdalabs/pokemon-blip-captions" \
    --cache_dir="original_dataset" \
    --test_size=200 \
    --seed=2137 \
    --output_dir="data/pokemon-split-200-train"