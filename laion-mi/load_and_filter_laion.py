from img2dataset import download
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download

from config import AESTHETIC_OUTPUT_DIR, AESTHETIC_OUTPUT_IMGS_DIR
from utils import download_from_urls

PARQUET_FILENAME = "part-00001-66bc9e17-4cc9-4c76-aaed-af85e99e94d2-c000.snappy.parquet"
REPO_NAME = "laion/laion2B-multi-joined-translated-to-en"
OUTPUT_DIR = "laion2b_multi_test_raw_2"
AESTHETIC_URLS = "urls.parquet"
AESTHETIC_THRS = 5
AESTHETIC_SCORE_COL = "prediction"
IS_AESTHETIC = (
    lambda x: x[AESTHETIC_SCORE_COL] > AESTHETIC_THRS
    if x[AESTHETIC_SCORE_COL] is not None
    else False
)

CAPTION_COL = "ENG TEXT"
COLUMNS = ["URL", CAPTION_COL, AESTHETIC_SCORE_COL]
SEED = 4321
IMGS_TO_SAVE = 300_000


def get_indices(data: Dataset) -> np.ndarray:
    if data.shape[0] <= IMGS_TO_SAVE:
        return np.arange(data.shape[0])
    return np.random.choice(range(data.shape[0]), IMGS_TO_SAVE, replace=False)


def save_metadata(aesthetic_data: Dataset):
    np.random.seed(SEED)
    indices = get_indices(aesthetic_data)
    trunc_aesthetic_data = aesthetic_data.select(indices)
    trunc_aesthetic_data.save_to_disk(AESTHETIC_OUTPUT_DIR)
    return trunc_aesthetic_data


def save_urls_to_parquet(data: Dataset):
    urls = pa.table([data[col] for col in COLUMNS], names=COLUMNS)
    pq.write_table(urls, os.path.join(AESTHETIC_OUTPUT_DIR, AESTHETIC_URLS))


def main():
    print("script start")
    hf_hub_download(
        REPO_NAME, PARQUET_FILENAME, repo_type="dataset", cache_dir=OUTPUT_DIR
    )
    print("raw data downloaded")
    data = load_dataset(OUTPUT_DIR)
    print("dataset setup")
    aesthetic_data = data["train"].filter(IS_AESTHETIC)
    print("dataset filtered", aesthetic_data.shape)
    aesthetic_data = save_metadata(aesthetic_data)
    print("metadata saved")
    save_urls_to_parquet(aesthetic_data)
    print("urls saved")
    download_from_urls(
        urls_parquet_path=os.path.join(AESTHETIC_OUTPUT_DIR, "urls.parquet"),
        output_path=AESTHETIC_OUTPUT_IMGS_DIR,
        url_col="URL",
        additional_colums=[AESTHETIC_SCORE_COL],
        caption_col=CAPTION_COL,
    )
    print("images downloaded")


if __name__ == "__main__":
    main()
