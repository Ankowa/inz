import os
import cv2

# from img2dataset import download
from typing import List, Tuple, Dict

import pyarrow as pa
import pyarrow.compute as pc
import torch

import torchvision
from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


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
        url_fixer: Dict[str, str] = None,
    ):
        assert (
            preprocess is None or "jpg" in columns
        ), "No jpg in columns with preprocess"
        if not "jpg" in columns:
            self.data = pq_table.select(columns)
        else:
            self.data = pq_table.filter(pc.equal(pq_table["status"], "success")).select(
                columns
            )
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
