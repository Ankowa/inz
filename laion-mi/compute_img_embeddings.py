from img2dataset import download
import clip
import torch
import os

from tqdm import tqdm

import numpy as np

from PIL import Image
from torch.utils.data import DataLoader

from torchvision.models import resnet50, ResNet50_Weights

import pyarrow as pa
import pyarrow.parquet as pq

# from config import AESTHETIC_OUTPUT_IMGS_DIR

OUTPUT_IMGS_DIR = "out/img_embs"
MODEL_TYPE = "clip"

from utils import PQDataset, load_parquet_files
from typing import List, Tuple

BATCH_SIZE = 500

device = "cuda:1"


def get_model_transforms():
    if MODEL_TYPE == "clip":
        model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
        model.to(device)
    elif MODEL_TYPE == "resnet":
        weights = ResNet50_Weights.IMAGENET1K_V2
        preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

        model = resnet50(weights=weights).to(device)

    return model, preprocess


def collate_candidates(sample) -> Tuple[torch.Tensor, List[str], List[str]]:
    # example[2] == is_ok flag
    imgs = torch.cat(
        [example[0].unsqueeze(0) for example in sample if example[2]], dim=0
    )
    urls = [example[1]["url"] for example in sample if example[2]]
    captions = [example[1]["caption"] for example in sample if example[2]]
    return imgs, urls, captions


def get_batch_emb(img, model):
    with torch.no_grad():
        if MODEL_TYPE == "clip":
            image_emb = model.encode_image(img).to("cpu")
        elif MODEL_TYPE == "resnet":
            image_emb = model(img).to("cpu")
        image_emb /= image_emb.norm(dim=-1, keepdim=True)
        return image_emb


def get_embeddings_metadata(
    loader: DataLoader,
    model,
) -> Tuple[torch.Tensor, pa.lib.Table]:
    embeddings, urls, captions = [], [], []
    for batch in tqdm(loader):
        imgs, urls_tmp, captions_tmp = batch
        imgs = imgs.to(device)
        emb_batch = get_batch_emb(imgs, model)
        embeddings.append(emb_batch)
        urls += urls_tmp
        captions += captions_tmp
        torch.cuda.empty_cache()

    metadata = pa.table([urls, captions], names=["url", "caption"])
    return torch.cat(embeddings, dim=0), metadata


def save_embeddings_metadata(embeddings: torch.Tensor, metadata: pa.lib.Table):
    with open(
        os.path.join(OUTPUT_IMGS_DIR, f"{MODEL_TYPE}_embeddings.pt"), "wb"
    ) as f_obj:
        torch.save(embeddings, f_obj)
    pq.write_table(
        metadata, os.path.join(OUTPUT_IMGS_DIR, f"{MODEL_TYPE}_metadata.parquet")
    )


def load_parquets(directory: str, files_cnt_to_take=np.inf):
    files = load_parquet_files(directory)
    table = []
    for idx, file in enumerate(files):
        if "metadata" in file:
            continue
        table.append(pq.read_table(file))
        if idx >= files_cnt_to_take:
            break
    table = pa.concat_tables(table)
    return table


def main():
    print("Loading model and transforms...")
    model, preprocess = get_model_transforms()
    print("Loading parquet files...")
    table = load_parquets(OUTPUT_IMGS_DIR, files_cnt_to_take=50)
    print("Creating dataset...")
    dataset = PQDataset(table, columns=["url", "caption", "jpg"], preprocess=preprocess)
    print("Creating loader...")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_candidates,
        num_workers=4,
    )
    print("Getting embeddings and metadata...")
    embeddings, metadata = get_embeddings_metadata(loader, model)
    print("Saving embeddings and metadata...")
    save_embeddings_metadata(embeddings, metadata)


if __name__ == "__main__":
    main()
