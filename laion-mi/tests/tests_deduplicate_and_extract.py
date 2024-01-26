from deduplicate_and_extract import (
    get_output_folder,
    get_l2_dir,
    get_non_members_folder,
    unique_non_members_extraction,
)
from config import (
    CHECKPOINT,
    L2_DISTANCE_THRESHOLD,
    SEARCH_QUERY_IMGS_DIR,
    L2_DISTANCES_DIR,
    NON_MEMBERS_IMG_DIR,
)


def test_get_output_folder():
    assert get_output_folder() == SEARCH_QUERY_IMGS_DIR

def test_get_l2_dir():
    assert get_l2_dir() == L2_DISTANCES_DIR

def test_get_non_members_folder():
    assert get_non_members_folder() == NON_MEMBERS_IMG_DIR

def test_unique_non_members_extraction():
    distances = {
        "img1": {
            "img2": 0.1,
            "img3": 0.2,
        },
        "img2": {
            "img1": 0.1,
            "img3": 0.3,
        },
        "img3": {
            "img1": 0.2,
            "img2": 0.3,
        },
    }
    df = unique_non_members_extraction(distances)
    assert df.shape == (2, 3)
    assert df.columns == ["img1", "img2", "dist"]
    assert df.loc[0]["img1"] == "img1"
    assert df.loc[0]["img2"] == "img3"
    assert df.loc[0]["dist"] == 0.2
    assert df.loc[1]["img1"] == "img2"
    assert df.loc[1]["img2"] == "img3"
    assert df.loc[1]["dist"] == 0.3
    