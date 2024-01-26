import torch
import os
import json

import numpy as np

from huggingface_hub import hf_hub_download
from datasets import load_dataset, load_from_disk

from tqdm import tqdm, trange
import pyarrow.parquet as pq
import pyarrow as pa
from torch.utils.data import DataLoader, Dataset, Subset
from utils import PQDataset, OnlyMembersDataset, MembershipDataset
from transformers import CLIPTextModel, CLIPTokenizer
from sklearn.metrics import confusion_matrix

import torch.nn as nn
import torch.optim as optim

from typing import Tuple, List

PARQUET_FILENAME = "5/2B-en-5_1.parquet"
REPO_NAME = "ChristophSchuhmann/improved_aesthetics_5plus"
OUTPUT_DIR = "laion2b_en_raw"
AESTHETIC_MEMBERS_DIR = "laion2b_members_aesthetic"
AESTHETIC_THRS = 5
AESTHETIC_SCORE_COL = "AESTHETIC_SCORE"
IS_AESTHETIC = lambda x: (
    x[AESTHETIC_SCORE_COL] > AESTHETIC_THRS
    if x[AESTHETIC_SCORE_COL] is not None
    else False
)
NON_MEMBERS_DIR = "laion2b_multi_bigger_non_members_imgs_checkpoint_24000_embeddings"
M_SET_DIR = "laion2b_members_iteratively_sanitized"


TEXT_ENCODER_BATCH_SIZE = 1_000
CLF_BATCH_SIZES = [
    10_000,
    2_000,
    128,
]
CHECKPOINT = 24_000
MEM_BATCH_SIZE = 40_000
EMBEDDING_SIZE = 77 * 768
EPOCHS = 100
EVAL_EVERY = 2

device = "cuda:0"

tokenizer = CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="tokenizer",
)

text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="text_encoder",
)
# text_encoder.to(device)

MODELS = [
    nn.Sequential(
        nn.Linear(EMBEDDING_SIZE, 1),
        nn.Sigmoid(),
    ),
    nn.Sequential(
        nn.Linear(EMBEDDING_SIZE, 4096),
        nn.LayerNorm(4096),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(4096, 1),
        nn.Sigmoid(),
    ),
    nn.Sequential(
        nn.Linear(EMBEDDING_SIZE, 8192),
        nn.LayerNorm(8192),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(8192, 512),
        nn.LayerNorm(512),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid(),
    ),
]

ITERATIONS = 3


def tokenize(example):
    try:
        tokens = tokenizer(
            [example],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    except ValueError:
        tokens = tokenizer(
            [""],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    return tokens.input_ids


def collate_prompts(sample):
    # example[0] is img which we don't load here, example[2] is is_ok which is always true
    urls = [example[1]["url"] for example in sample]
    tokens = torch.cat([tokenize(example[1]["caption"]) for example in sample], dim=0)
    prompts = [example[1]["caption"] for example in sample]
    try:
        indices = [example[1]["idx"] for example in sample]
    except KeyError:
        indices = None
    return urls, tokens, prompts, indices


def members_collate(sample):
    embeddings = torch.cat([example[0] for example in sample], dim=0)
    metadata = pa.concat_tables([example[1] for example in sample])
    return embeddings, metadata


def get_members_data() -> pa.lib.Table:
    try:
        members = load_from_disk(AESTHETIC_MEMBERS_DIR)
    except FileNotFoundError:
        hf_hub_download(
            REPO_NAME, PARQUET_FILENAME, repo_type="dataset", cache_dir=OUTPUT_DIR
        )
        print("raw data downloaded")
        members = load_dataset(OUTPUT_DIR)["train"].filter(IS_AESTHETIC)
        members.save_to_disk(AESTHETIC_MEMBERS_DIR)
    return members.data.table.select(
        ["URL", "TEXT", "__index_level_0__"]
    ).rename_columns(["url", "caption", "idx"])


def get_non_members_data() -> Tuple[List[str], List[str], torch.Tensor]:
    with open(os.path.join(NON_MEMBERS_DIR, "embeddings.pt"), "rb") as f_obj:
        non_members_embeddings = torch.load(f_obj)
    with open(os.path.join(NON_MEMBERS_DIR, "urls.json"), "r") as f_obj:
        urls = json.load(f_obj)
    with open(os.path.join(NON_MEMBERS_DIR, "prompts.json"), "r") as f_obj:
        captions = json.load(f_obj)

    return (
        urls,
        captions,
        non_members_embeddings.view(non_members_embeddings.shape[0], -1),
    )


def generate_members_batch_indices(members_cnt) -> np.ndarray:
    np.random.seed(123)
    indices = np.random.permutation(members_cnt)
    for batch in np.array_split(indices, members_cnt // MEM_BATCH_SIZE):
        yield batch


@torch.no_grad()
def get_members_embeddings_data(
    members_batch: pa.lib.Table,
    encoder: CLIPTextModel = text_encoder,
    device: str = device,
) -> Tuple[torch.Tensor, pa.lib.Table]:
    ds = PQDataset(members_batch, columns=["url", "caption", "idx"])
    loader = DataLoader(
        ds,
        batch_size=TEXT_ENCODER_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_prompts,
        num_workers=4,
    )
    urls, embeddings, captions, indices = [], [], [], []
    for batch in tqdm(loader):
        # print(batch)
        urls_batch, tokens, captions_batch, indices_batch = batch
        urls += urls_batch
        captions += captions_batch
        indices += indices_batch
        out = encoder(tokens.to(device))[0]
        embeddings.append(out.cpu().view(out.shape[0], -1))
        torch.cuda.empty_cache()

    embeddings = torch.cat(embeddings, dim=0)
    metadata = pa.table([urls, captions, indices], names=["url", "caption", "idx"])
    return embeddings, metadata


def save_iteration_data(
    m_i: pa.lib.Table, model: nn.Sequential, iteration: int
) -> None:
    datapath = os.path.join(M_SET_DIR, str(iteration))
    os.makedirs(datapath, exist_ok=True)
    with open(os.path.join(datapath, "model.pt"), "wb") as f_obj:
        torch.save(model, f_obj)
    pq.write_table(m_i, os.path.join(datapath, "members.parquet"))


def get_iteration_data(iteration: int) -> pa.lib.Table:
    datapath = os.path.join(M_SET_DIR, str(iteration))
    members = pq.read_table(os.path.join(datapath, "members.parquet"))
    return members


def get_iteration_model(iteration: int) -> nn.Sequential:
    datapath = os.path.join(M_SET_DIR, str(iteration))
    with open(os.path.join(datapath, "model.pt"), "rb") as f_obj:
        model = torch.load(f_obj)
    return model


def get_train_test_loaders(
    dataset: MembershipDataset, indices: np.ndarray, train_size: float, iteration: int
) -> Tuple[DataLoader, DataLoader]:
    torch.manual_seed(123)
    train_data = Subset(dataset, indices[: int(len(indices) * train_size)])
    test_data = Subset(dataset, indices[int(len(indices) * train_size) :])
    train_loader = DataLoader(
        train_data,
        batch_size=CLF_BATCH_SIZES[iteration],
        pin_memory=True,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=CLF_BATCH_SIZES[iteration],
        pin_memory=True,
        shuffle=True,
        num_workers=4,
    )
    return train_loader, test_loader


def evaluate(
    model: nn.Sequential,
    loader: DataLoader,
    loss_func=nn.BCELoss(),
    device: str = device,
) -> float:
    with torch.no_grad():
        model.eval()
        loss = 0
        cnts = 0
        for data in loader:
            X = data[0].to(device)
            y = data[1].to(device)
            y_hat = model(X)
            loss += loss_func(y_hat, y).item()
            cnts += len(X)
        return loss / cnts


def train(
    model: nn.Sequential,
    loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    eval_every: int,
    optimizer: torch.optim.Optimizer,
    loss_func=nn.BCELoss(),
    device: str = device,
):
    model.train()
    valid_loss_prev = np.inf
    with trange(epochs) as tepochs:
        for epoch in tepochs:
            torch.cuda.empty_cache()
            running_loss = 0
            for data in loader:
                X = data[0].to(device)
                y = data[1]
                y_hat = model(X).cpu()
                loss = loss_func(y_hat, y)
                running_loss += loss.item() / len(X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if not (epoch + 1) % eval_every:
                valid_loss = evaluate(model, eval_loader, device=device)
                tepochs.set_postfix(
                    valid_loss={round(valid_loss, 7)},
                    train_loss={round(running_loss, 7)},
                )
                if valid_loss > valid_loss_prev:
                    print("Early stopping")
                    break
                valid_loss_prev = valid_loss


def extract_fps(
    model: nn.Sequential,
    loader: DataLoader,
    return_embeddings: bool = False,
    device: str = device,
) -> pa.lib.Table:
    with torch.no_grad():
        model.eval()
        metadata = []
        if return_embeddings:
            embeddings = []
        model.to(device)
        for data in loader:
            X = data[0].to(device)
            y_pred = model(X).cpu().view(-1)
            metadata.append(data[1].take(np.arange(len(y_pred))[y_pred > 0.5]))
            if return_embeddings:
                embeddings.append(X[y_pred > 0.5].cpu())
        model.cpu()
        torch.cuda.empty_cache()
        metadata = pa.concat_tables(metadata)
        if return_embeddings:
            embeddings = torch.cat(embeddings, dim=0)
            return metadata, embeddings
        return metadata


def extract_fps_iterative(
    iteration: int, loader: DataLoader, iteration_model: nn.Sequential
) -> pa.lib.Table:
    with torch.no_grad():
        iteration_model.eval()
        for iteration in range(iteration):
            model = get_iteration_model(iteration)
            metadata, embeddings = extract_fps(model, loader, return_embeddings=True)
            loader = DataLoader(
                OnlyMembersDataset(embeddings, metadata),
                batch_size=CLF_BATCH_SIZES[iteration],
                pin_memory=True,
                shuffle=False,
                collate_fn=members_collate,
                num_workers=4,
            )
            del model
            torch.cuda.empty_cache()
        return extract_fps(iteration_model, loader)


def get_cm(
    model: nn.Sequential, loaders: List[DataLoader], device: str = device
) -> None:
    with torch.no_grad():
        model.eval()
        for loader in loaders:
            y_true, y_pred = [], []
            for data in loader:
                X = data[0].to(device)
                y = data[1].cpu()
                y_hat = model(X).cpu()
                y_true.append(y)
                y_pred.append(y_hat)
            y_true = torch.cat(y_true, dim=0).numpy()
            y_pred = torch.cat(y_pred, dim=0).numpy()
            return confusion_matrix(y_true, y_pred > 0.5, normalize="true")


def first_iteration(
    non_members_embeddings: torch.Tensor,
    members_data: pa.lib.Table,
    members_indices_generator,
):
    members_embeddings, members_metadata = get_members_embeddings_data(
        members_data.take(next(members_indices_generator))
    )
    dataset = MembershipDataset(members_embeddings, non_members_embeddings)

    np.random.seed(42)
    train_loader, test_loader = get_train_test_loaders(
        dataset, np.random.permutation(len(dataset)), 0.8, 0
    )
    model = MODELS[0].to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, test_loader, EPOCHS, EVAL_EVERY, optimizer)
    print(get_cm(model, [train_loader, test_loader]))

    model.cpu()

    members_final_metadata = list()
    for batch in members_indices_generator:
        members_embeddings, members_metadata = get_members_embeddings_data(
            members_data.take(batch)
        )
        members_ds = OnlyMembersDataset(members_embeddings, members_metadata)
        members_loader = DataLoader(
            members_ds,
            batch_size=CLF_BATCH_SIZES[0],
            pin_memory=True,
            shuffle=False,
            collate_fn=members_collate,
            num_workers=4,
        )
        members_final_metadata.append(extract_fps(model, members_loader))
        fp_members = sum([metadata.shape[0] for metadata in members_final_metadata])
        print("FP members extracted so far:", fp_members)
        if fp_members > MEM_BATCH_SIZE:
            break
    print("members final metadata computed")
    members_final_metadata = pa.concat_tables(members_final_metadata)
    save_iteration_data(members_final_metadata, model, 0)


def n_th_iteration(
    non_members_embeddings: torch.Tensor,
    members_data: pa.lib.Table,
    iteration: int,
    members_indices_generator,
):
    members_embeddings, members_metadata = get_members_embeddings_data(
        get_iteration_data(iteration - 1)
    )
    dataset = MembershipDataset(members_embeddings, non_members_embeddings)
    np.random.seed(42)
    train_loader, test_loader = get_train_test_loaders(
        dataset, np.random.permutation(len(dataset)), 0.8, iteration
    )
    model = MODELS[iteration].to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    loss_func = nn.BCELoss()
    train(model, train_loader, test_loader, EPOCHS, EVAL_EVERY, optimizer, loss_func)
    print(get_cm(model, [train_loader, test_loader]))

    model.cpu()

    members_final_metadata = list()
    for batch in members_indices_generator:
        members_embeddings, members_metadata = get_members_embeddings_data(
            members_data.take(batch)
        )
        members_ds = OnlyMembersDataset(members_embeddings, members_metadata)
        members_loader = DataLoader(
            members_ds,
            batch_size=CLF_BATCH_SIZES[iteration],
            pin_memory=True,
            shuffle=False,
            collate_fn=members_collate,
            num_workers=4,
        )
        members_final_metadata.append(
            extract_fps_iterative(iteration, members_loader, model)
        )
        fp_members = sum([metadata.shape[0] for metadata in members_final_metadata])
        print("FP members extracted so far:", fp_members)
        if fp_members > MEM_BATCH_SIZE:
            break
    print("members final metadata computed")
    members_final_metadata = pa.concat_tables(members_final_metadata)
    save_iteration_data(members_final_metadata, model, iteration)


def main():
    members = get_members_data()
    print("members loaded", members.shape)
    _, _, non_members = get_non_members_data()
    print("non-members loaded", non_members.shape)
    members_indices_generator = generate_members_batch_indices(members.shape[0])

    print("starting first iteration")
    first_iteration(non_members, members, members_indices_generator)
    for iteration in range(1, ITERATIONS):
        print("iteration", iteration)
        n_th_iteration(non_members, members, iteration, members_indices_generator)


if __name__ == "__main__":
    main()
