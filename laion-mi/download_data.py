from utils import download_from_urls
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--url_col", type=str, default="url", required=False)
    parser.add_argument("--additional_colums", type=str, required=False)
    parser.add_argument("--caption_col", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_from_urls(
        urls_parquet_path=args.urls,
        output_path=args.output_path,
        url_col=args.url_col,
        additional_colums=args.additional_colums,
        caption_col=args.caption_col,
    )
