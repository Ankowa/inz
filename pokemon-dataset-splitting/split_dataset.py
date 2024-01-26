import os
import argparse
import datasets
from typing import Union


def parse_args():
    parser = argparse.ArgumentParser(
        description="simple script to spit out train_test_split'ed dataset"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). "
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="original_dataset",
        required=False,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--test_size",
        default=200,
        required=False,
        help="test size, if int then rows, if float then ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2137,
        required=False,
        help="random split seed, for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pokemon-split",
        help="The output directory where the new dataset will be saved",
    )
    args = parser.parse_args()
    try:
        args.test_size = int(args.test_size)
    except ValueError as e:
        args.test_size = float(args.test_size)
        assert args.test_size >= 0 and args.test_size <= 1
    return args


def main():
    args = parse_args()
    dataset = datasets.load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    dataset = dataset["train"].train_test_split(
        test_size=args.test_size, seed=args.seed
    )
    dataset.save_to_disk(os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
