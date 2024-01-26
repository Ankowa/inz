from typing import List
from img2dataset import download

URLS = "members-raw.parquet"
OUTPUT_PATH = "members-raw"


def download_from_urls(
    urls_parquet_path: str,
    output_path: str,
    url_col: str,
    additional_colums: List[str],
    caption_col: str = None,
) -> None:
    download(
        processes_count=16,
        thread_count=32,
        url_list=urls_parquet_path,
        resize_mode="no",
        output_folder=output_path,
        disable_all_reencoding=True,
        output_format="files",
        input_format="parquet",
        url_col=url_col,
        caption_col=caption_col,
        save_additional_columns=additional_colums,
        enable_wandb=False,
        number_sample_per_shard=1000,
        distributor="multiprocessing",
    )


if __name__ == "__main__":
    download_from_urls(
        urls_parquet_path=URLS,
        output_path=OUTPUT_PATH,
        url_col="url",
        additional_colums=None,
        caption_col="caption",
    )
