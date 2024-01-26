import json
import pyarrow.parquet as pq
from utils import download_from_urls

import os
import pandas as pd
from config import AESTHETIC_OUTPUT_DIR, L2_DISTANCES_DIR, NON_MEMBERS_DATASET_DIR


def load_raw():
    original_dataset = pq.read_table(
        os.path.join(AESTHETIC_OUTPUT_DIR, "urls.parquet")
    ).to_pandas()
    with open(os.path.join(L2_DISTANCES_DIR, "non-members-urls.json"), "r") as f_obj:
        non_members_urls = json.load(f_obj)
    return original_dataset, non_members_urls


def process_and_save_metadata(
    original_dataset: pd.DataFrame, non_members_urls: list
) -> None:
    os.makedirs(NON_MEMBERS_DATASET_DIR, exist_ok=True)
    original_dataset.loc[original_dataset.URL.isin(non_members_urls)].to_parquet(
        os.path.join(NON_MEMBERS_DATASET_DIR, "metadata.parquet")
    )


def main():
    original_dataset, non_members_urls = load_raw()
    process_and_save_metadata(original_dataset, non_members_urls)
    download_from_urls(
        urls_parquet_path=os.path.join(NON_MEMBERS_DATASET_DIR, "metadata.parquet"),
        output_path=NON_MEMBERS_DATASET_DIR,
        url_col="URL",
        additional_colums=["prediction"],
        caption_col="ENG TEXT",
    )


if __name__ == "__main__":
    main()
