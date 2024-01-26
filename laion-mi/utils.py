import os
import cv2
from img2dataset import download
from typing import List, Tuple, Dict, Callable
from multiprocessing import Process
import threading
import json

import pyarrow as pa
import pyarrow.compute as pc
import torch

import torchvision
from torch.utils.data import Dataset

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from config import CHECKPOINT_DIR
import matplotlib.pyplot as plt

import numpy as np


def download_from_urls(
    urls_parquet_path: str,
    output_path: str,
    url_col: str,
    additional_colums: List[str],
    caption_col: str = "",
    **kwargs,
) -> None:
    download(
        processes_count=16,
        thread_count=32,
        url_list=urls_parquet_path,
        resize_mode="no",
        output_folder=output_path,
        disable_all_reencoding=True,
        output_format="parquet",
        input_format="parquet",
        url_col=url_col,
        caption_col=caption_col,
        save_additional_columns=additional_colums,
        enable_wandb=False,
        number_sample_per_shard=1000,
        distributor="multiprocessing",
        **kwargs,
    )


def load_parquet_files(dataset_path: str) -> List[str]:
    files = []
    for file in os.listdir(dataset_path):
        if not file.endswith(".parquet"):
            continue
        files.append(os.path.join(dataset_path, file))
    return files


class PQDataset(Dataset):
    def __init__(
        self,
        pq_table: pa.lib.Table,
        preprocess: torchvision.transforms.Compose = None,
        columns=["url_orig", "jpg"],
        url_fixer: Dict[str, str] = dict(),
    ):
        assert (
            preprocess is None or "jpg" in columns
        ), "No jpg in columns with preprocess"
        if not "jpg" in columns:
            self.data = pq_table.select(columns)
        else:
            success = pc.field("status") == "success"
            self.data = pq_table.filter(success).select(columns)
        self.columns = columns
        self.transform = preprocess
        self.url_fixer = url_fixer

    def img_to_tensor(self, img: bytes) -> Tuple[torch.Tensor, bool]:
        try:
            image = Image.open(BytesIO(img))
            return self.transform(image), True
        except:  # can fail if bytes have some weird headers
            try:
                image = cv2.imdecode(
                    np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR
                )  # can fail if bytes empty
                image = cv2.cvtColor(
                    image, cv2.COLOR_BGR2RGB
                )  # can fail if bytes are e.g. HTML for some reason
                image = Image.fromarray(image)
                return self.transform(image), True
            except:  # if it fails we return an empty tensor and is_ok = False
                return torch.tensor([]), False

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict, bool]:
        row = self.data.slice(idx, length=1)
        if "jpg" in self.columns:
            img, is_ok = self.img_to_tensor(row.select(["jpg"]).to_pylist()[0]["jpg"])
        else:
            img, is_ok = None, True
        metadata = row.select(
            [col for col in self.columns if col != "jpg"]
        ).to_pylist()[0]
        if self.url_fixer is not None and "url_orig" in self.columns:
            metadata["url_orig"] = self.url_fixer[
                metadata["url_orig"]
            ]  # bug cropped urls by one letter too much
        return img, metadata, is_ok


def get_url_fixer(correct_urls: List[str]) -> Dict[str, str]:
    return {url[:-1]: url for url in correct_urls}


class MembershipDataset(Dataset):
    def __init__(self, members, non_members):
        self.X = torch.cat([members, non_members], dim=0)
        self.y = torch.cat(
            [
                torch.zeros((members.shape[0], 1)),
                torch.ones((non_members.shape[0], 1)),
            ],
            dim=0,
        )

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class OnlyMembersDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, metadata: pa.lib.Table):
        self.embeddings = embeddings
        self.metadata = metadata

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx].unsqueeze(0), self.metadata.slice(idx, length=1)


class QueryThread(threading.Thread):
    def __init__(
        self,
        procID: int,
        threadID: int,
        embeddings: List[torch.Tensor],
        urls: List[str],
        search_client: Callable,
        get_response: Callable,
        checkpoint_every: int = 1000,
    ):
        threading.Thread.__init__(self)
        self.procID = procID
        self.threadID = threadID
        self.results: list = []
        self.embeddings = embeddings
        self.urls = urls
        self.search_client = search_client()
        self.get_response = get_response
        self.checkpoint_every = checkpoint_every

    def run(self) -> None:
        idx = 0
        for url, embedding in zip(
            tqdm(self.urls, desc=f"P: {self.procID}, t: {self.threadID}"),
            self.embeddings,
        ):
            try:
                result = self.get_response(embedding, url, self.search_client)
            except Exception as e:
                print(self.threadID, e)
                continue
            self.results.append(result)
            idx += 1
            if not idx % self.checkpoint_every:
                self.checkpoint_save(idx)

    def checkpoint_save(self, idx):
        path = os.path.join(
            CHECKPOINT_DIR, f"thread_{self.threadID}", f"checkpoint_{idx}"
        )
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "data.json"), "w") as f_obj:
            json.dump(self.results, f_obj)


class QueryProcess(Process):
    def __init__(
        self,
        procID: int,
        embeddings: List[torch.Tensor],
        urls: List[str],
        threads_cnt: int,
        create_threads: Callable,
        run_threads: Callable,
    ):
        Process.__init__(self)
        self.procID = procID
        self.results: list = list()
        self.embeddings = embeddings
        self.urls = urls
        self.threads = create_threads(threads_cnt, embeddings, urls, procID)
        self.run_threads = run_threads

    def run(self):
        self.run_threads(self.threads)
        for thread in self.threads:
            self.results += thread.results


def setup_matplotlib():
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": 15,  # Set font size to 11pt
            "axes.labelsize": 15,  # -> axis labels
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2,
            "text.usetex": False,
            "pgf.rcfonts": False,
        }
    )
