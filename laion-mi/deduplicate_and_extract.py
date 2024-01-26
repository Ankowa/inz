import os
import json

from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader

import clip
import torch
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from typing import List

BATCH_SIZE = 800

from config import (
    SEARCH_QUERY_RESULTS_DIR,
    SEARCH_QUERY_IMGS_DIR,
    AESTHETIC_OUTPUT_IMGS_DIR,
    L2_DISTANCES_DIR,
    CHECKPOINT_DIR,
    NON_MEMBERS_IMG_DIR,
)

# PID: 15735

from utils import download_from_urls, get_url_fixer, load_parquet_files, PQDataset

L2_DISTANCE_THRESHOLD = 0.5
CHECKPOINT = 24_000

from query_knn_index import get_embeddings_urls

device = "cuda:1"

l2_score = lambda x, y: (x - y).pow(2).sum(-1).sqrt().item()

_, urls = get_embeddings_urls()
fix_url = get_url_fixer(urls)


def compute_checkpoint_pq_table():
    threads_dirs = os.listdir(CHECKPOINT_DIR)
    data = []
    for thread_dir in threads_dirs:
        tmp_path = os.path.join(CHECKPOINT_DIR, thread_dir, f"checkpoint_{CHECKPOINT}")
        with open(os.path.join(tmp_path, "data.json"), "r") as f_obj:
            datapart = json.load(f_obj)
            data += datapart

    final_results = {
        "url_orig": [],
        "url_sim": [],
    }
    for results in data:
        if len(results["results"]) == 0:
            new_results = {
                "url_orig": [fix_url[results["url"]]],
                "url_sim": ["NO DUPLICATES"],
            }
        else:
            new_results = {
                "url_orig": [fix_url[results["url"]]] * len(results["results"]),
                "url_sim": [
                    results["results"][idx]["url_sim"]
                    for idx in range(len(results["results"]))
                ],
            }
        for key, values in new_results.items():
            final_results[key] += values
    final_results = pa.table(
        [values for values in final_results.values()],
        names=[key for key in final_results.keys()],
    )
    os.makedirs(SEARCH_QUERY_RESULTS_DIR, exist_ok=True)
    pq.write_table(
        final_results,
        os.path.join(
            SEARCH_QUERY_RESULTS_DIR,
            f"search_query_urls_checkpoint_{CHECKPOINT}.parquet",
        ),
    )


def get_urls_filepath() -> str:
    if CHECKPOINT > -1:
        try:
            _ = pq.read_table(
                os.path.join(
                    SEARCH_QUERY_RESULTS_DIR,
                    f"search_query_urls_checkpoint_{CHECKPOINT}.parquet",
                )
            )
        except FileNotFoundError as e:
            print(e, "computing")
            compute_checkpoint_pq_table()
        return os.path.join(
            SEARCH_QUERY_RESULTS_DIR,
            f"search_query_urls_checkpoint_{CHECKPOINT}.parquet",
        )
    return os.path.join(SEARCH_QUERY_RESULTS_DIR, "search_query_urls.parquet")


def get_output_folder():
    if CHECKPOINT > -1:
        return f"{SEARCH_QUERY_IMGS_DIR}_checkpoint_{CHECKPOINT}"
    return SEARCH_QUERY_IMGS_DIR


def get_l2_dir() -> str:
    l2_dir = (
        L2_DISTANCES_DIR
        if CHECKPOINT == -1
        else f"{L2_DISTANCES_DIR}_checkpoint_{CHECKPOINT}"
    )
    return l2_dir


def get_non_members_folder():
    if CHECKPOINT > -1:
        return f"{NON_MEMBERS_IMG_DIR}_checkpoint_{CHECKPOINT}"
    return NON_MEMBERS_IMG_DIR


def download_duplicate_candidates():
    url_list = get_urls_filepath()
    output_folder = get_output_folder()
    download_from_urls(
        url_list, output_folder, url_col="url_sim", additional_colums=["url_orig"]
    )


def collate_candidates(sample):
    # example[2] == is_ok flag
    img_batch = torch.cat(
        [example[0].unsqueeze(0) for example in sample if example[2]], dim=0
    )
    urls_orig = [example[1]["url_orig"] for example in sample if example[2]]
    urls_sim = [example[1]["url"] for example in sample if example[2]]
    return img_batch, urls_orig, urls_sim


def get_batch_emb(model, img):
    with torch.no_grad():
        image_emb = model.encode_image(img).to("cpu")
        image_emb /= image_emb.norm(dim=-1, keepdim=True)
        return image_emb.type(torch.float32)


def get_url_embeddings_dict_orig() -> dict:
    embeddings, urls = get_embeddings_urls()
    return {url: embedding for url, embedding in zip(urls, list(embeddings))}


def get_parquet_loader(
    parquet_file: str, preprocess: torchvision.transforms.Compose
) -> DataLoader:
    dup_candidates_table = pq.read_table(parquet_file)
    dataset = PQDataset(
        pq_table=dup_candidates_table,
        preprocess=preprocess,
        columns=["url_orig", "url", "jpg"],
        # url_fixer=fix_url,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_candidates
    )
    return loader


def get_l2_distances(
    parquet_files: List[str], orig_url_emb: dict(), model, preprocess
) -> dict:
    output = {url: dict() for url in orig_url_emb.keys()}
    for parquet_file in tqdm(parquet_files, desc="l2"):
        loader = get_parquet_loader(parquet_file, preprocess)  # memory
        for batch in loader:
            try:
                (
                    img_batch,
                    urls_orig,
                    urls_sim,
                ) = batch  # e.g. all images from the batch fail to load → torch.cat error
            except:
                continue
            img_batch = img_batch.to(device)
            emb_batch = get_batch_emb(model, img_batch)
            for idx_, embedding_sim in enumerate(list(emb_batch)):
                url_orig = urls_orig[idx_]
                url_sim = urls_sim[idx_]
                embedding_orig = orig_url_emb[url_orig]
                l2 = l2_score(embedding_sim, embedding_orig)
                output[url_orig][url_sim] = l2
    return output


def save_distances(l2_distances: dict):
    l2_dir = get_l2_dir()
    os.makedirs(l2_dir, exist_ok=True)
    with open(os.path.join(l2_dir, "distances.json"), "w") as f_obj:
        json.dump(l2_distances, f_obj)


def distances_computation() -> dict:
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    model.to(device)
    print("CLIP Loaded")
    parquet_files = load_parquet_files(get_output_folder())
    orig_url_embeddings = get_url_embeddings_dict_orig()
    print("original url-embeddings loaded", len(orig_url_embeddings))
    l2_distances = get_l2_distances(
        parquet_files, orig_url_embeddings, model, preprocess
    )
    save_distances(l2_distances)
    return l2_distances


def get_df_min(distances: dict) -> pd.DataFrame:
    orig_urls, sim_urls, dists = [], [], []
    for orig_url, data in distances.items():
        if len(data) == 0:
            """
            Here we've decided to not include samples for which we didn't get any l2 distances.
            It's because this can happen due to API failure, connection failure, wrong format
            of the duplicate candidates images or other unknown factors. To maintain full
            control over the deduplication process we skip these non-member candidates.
            """
            continue
        orig_urls += [orig_url]
        sim_url = min(distances[orig_url], key=lambda x: distances[orig_url][x])
        sim_urls += [sim_url]
        dists += [distances[orig_url][sim_url]]
    df = pd.DataFrame(
        {
            "url_orig": orig_urls,
            "url_sim": sim_urls,
            "dist": dists,
        }
    )
    return df


def unique_non_members_extraction(distances: dict):
    df = get_df_min(distances)
    df = df.loc[
        df.dist >= L2_DISTANCE_THRESHOLD
    ]  # only samples with the lowest L2 score to their duplicate candidates above L2_DISTANCE_THRESHOLD
    l2_dir = get_l2_dir()
    os.makedirs(l2_dir, exist_ok=True)
    with open(os.path.join(l2_dir, "non-members-urls.json"), "w") as f_obj:
        json.dump(df.url_orig.values.tolist(), f_obj)


def download_non_members():
    with open(os.path.join(get_l2_dir(), "non-members-urls.json"), "r") as f_obj:
        urls_to_save = json.load(f_obj)
    urls_path = os.path.join(get_l2_dir(), "non-members-urls.parquet")
    pq_files = load_parquet_files(AESTHETIC_OUTPUT_IMGS_DIR)
    url_caption_all = dict()
    for file in tqdm(pq_files, desc="pq"):
        table = pq.read_table(file)
        for row in table.to_pylist():
            url_caption_all[row["url"]] = row["caption"]

    url_caption = {url: url_caption_all[url] for url in urls_to_save}
    urls = [url for url in url_caption.keys()]
    captions = [caption for caption in url_caption.values()]
    pq.write_table(pa.table([urls, captions], names=["URL", "TEXT"]), urls_path)
    output_folder = get_non_members_folder()
    download_from_urls(
        urls_path,
        output_folder,
        url_col="URL",
        caption_col="TEXT",
        additional_colums=None,
    )


def main():
    download_duplicate_candidates()
    distances = distances_computation()
    unique_non_members_extraction(distances)
    download_non_members()


if __name__ == "__main__":
    main()
