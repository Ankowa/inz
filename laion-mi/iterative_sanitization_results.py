import torch
import os
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from iterative_sanitization import get_members_embeddings_data
from utils import setup_matplotlib
from transformers import CLIPTextModel, CLIPTokenizer
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

device = "cuda:1"

tokenizer = CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="tokenizer",
)

text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="text_encoder",
)

text_encoder.to(device)
SANITIZED_MEMBERS_PATH = "out/laion2b_members_iteratively_sanitized"
setup_matplotlib()


def get_nm_emb_text(source):
    nonmembers = pq.read_table(source)
    nonmembers = nonmembers.add_column(2, "idx", [np.arange(nonmembers.shape[0])])
    nonmembers = nonmembers.rename_columns(["url", "caption", "idx"])

    nm_embeddings, _ = get_members_embeddings_data(
        nonmembers,
        text_encoder,
        device,
    )
    return nm_embeddings


def get_m_emb_text(source):
    members = pq.read_table(source)

    m_embeddings, _ = get_members_embeddings_data(
        members,
        text_encoder,
        device,
    )
    return m_embeddings


def get_nm_m_mn_no_emb_img():
    with open("nonmembers_imgs/clip_embeddings.pt", "rb") as f:
        nm_emb_imgs = torch.load(f).type(torch.float32)

    with open("members_imgs/clip_embeddings.pt", "rb") as f:
        m_emb_imgs = torch.load(f).type(torch.float32)

    return (
        nm_emb_imgs,
        m_emb_imgs,
    )


def get_pca_df(pca_members, pca_nonmembers):
    pca_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "pc1": sample_[:, 0],
                    "pc2": sample_[:, 1],
                    "membership": [hue] * sample_.shape[0],
                }
            )
            for sample_, hue in zip(
                [pca_members, pca_nonmembers], ["members", "nonmembers"]
            )
        ],
        axis=0,
    )
    return pca_df


def get_single_plot(pca_df, name):
    p = sns.jointplot(
        data=pca_df,
        x="pc1",
        y="pc2",
        hue="membership",
        kind="scatter",
        marginal_kws={"common_norm": False, "fill": False},
        s=2,
        alpha=0.5,
    )
    p.set_axis_labels("PCA first component", "PCA second component")
    p.fig.subplots_adjust(top=0.9)
    p.fig.suptitle(f"2D PCA plot - {name}")
    plt.savefig(f"plots/single-{name}.png")
    plt.show()


def get_pca_plot(m_emb, nm_emb, title):
    pca = IncrementalPCA(n_components=2)
    pca_data = np.concatenate([m_emb, nm_emb])

    for batch in tqdm(np.array_split(pca_data, 1000)):
        pca.partial_fit(batch)
    m_pca = pca.transform(m_emb)
    nm_pca = pca.transform(nm_emb)
    pca_df = get_pca_df(m_pca, nm_pca)

    get_single_plot(pca_df, title)


def main():
    m_emb = get_m_emb_text(os.path.join(SANITIZED_MEMBERS_PATH, "2", "members.parquet"))
    nm_emb = get_nm_emb_text(
        "out/laion_2b_multi_l2_distances_checkpoint_24000/non-members-urls.parquet"
    )
    get_pca_plot(m_emb, nm_emb, "prompts iteration 3")
    m_emb = get_m_emb_text("out/members-raw.parquet")
    get_pca_plot(m_emb, nm_emb, "prompts before sanitization")
    nm_emb_imgs, m_emb_imgs = get_nm_m_mn_no_emb_img()
    get_pca_plot(m_emb_imgs, nm_emb_imgs, "images iteration 3")
