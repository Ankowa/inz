from get_non_members import process_and_save_metadata
import os
import pandas as pd

def test_process_and_save_metadata():
    original_dataset = pd.DataFrame(
        {
            "URL": ["img1", "img2", "img3"],
            "caption": ["caption1", "caption2", "caption3"],
        }
    )
    non_members_urls = ["img1", "img3"]
    process_and_save_metadata(original_dataset, non_members_urls)
    assert os.path.exists("out/data/non_members/metadata.parquet")
    df = pd.read_parquet("out/data/non_members/metadata.parquet")
    assert df.shape == (2, 2)
    assert df.columns == ["URL", "caption"]
    assert df.URL.tolist() == ["img1", "img3"]
    assert df.caption.tolist() == ["caption1", "caption3"]