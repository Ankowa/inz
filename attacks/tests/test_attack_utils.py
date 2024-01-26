import pytest
from attacks.utils import PQDataset
import torch
from argparse import Namespace

from attacks.attack_utils import (
    get_sd_from_checkpoint,
    get_parquet_dataset,
    get_huggingface_dataset,
    get_dataset,
    get_l2_loss,
    get_image_loss,
    get_loss_shape,
)


@pytest.mark.parametrize(
    "checkpoint",
    [
        "5000",
        "10000",
        "15000",
        "20000",
        "25000",
        "30000",
    ],
)
def test_get_sd_from_checkpoint(checkpoint):
    args = Namespace(
        model_name="sd-finetuned",
        checkpoint_id=checkpoint,
    )
    sd = get_sd_from_checkpoint(args)
    assert isinstance(sd, dict)
    assert "model_state_dict" in sd
    assert "optimizer_state_dict" in sd
    assert "scheduler_state_dict" in sd


def test_get_parquet_dataset():
    data_path = "out/data/members.parquet"
    transform = None
    dataset = get_parquet_dataset(data_path, transform)
    assert isinstance(dataset, PQDataset)
    assert len(dataset) == 31783
    assert dataset.columns == ["url", "caption", "jpg"]


def test_get_huggingface_dataset():
    dataset = get_huggingface_dataset(
        "pokemon-split",
        None,
        True,
    )
    assert len(dataset) == 633
    assert dataset.features["jpg"].dtype == "int64"
    assert dataset.features["caption"].dtype == "string"
    assert dataset.features["jpg"].shape == (None, 3, 224, 224)
    assert dataset.features["caption"].shape == (None,)


def test_get_dataset():
    dataset = get_dataset(
        "pokemon-split",
        None,
        True,
    )
    assert len(dataset) == 633
    assert dataset.features["jpg"].dtype == "int64"
    assert dataset.features["caption"].dtype == "string"
    assert dataset.features["jpg"].shape == (None, 3, 224, 224)
    assert dataset.features["caption"].shape == (None,)


def test_get_loss_shape():
    loss = torch.tensor([1, 2, 3, 4, 5])
    assert get_loss_shape(loss, True).shape == (1, 1, 5)
    assert get_loss_shape(loss, False).shape == (1, 5)
    loss = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert get_loss_shape(loss, True).shape == (2, 1, 5)
    assert get_loss_shape(loss, False).shape == (2, 5)
    loss = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    assert get_loss_shape(loss, True).shape == (1, 2, 5)
    assert get_loss_shape(loss, False).shape == (1, 2, 5)


def test_get_l2_loss():
    true = torch.tensor([1, 2, 3, 4, 5])
    pred = torch.tensor([1, 2, 3, 4, 5])
    assert get_l2_loss(true, pred, True).shape == (1, 1, 5)
    assert get_l2_loss(true, pred, False).shape == (1, 5)
    true = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    pred = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert get_l2_loss(true, pred, True).shape == (2, 1, 5)
    assert get_l2_loss(true, pred, False).shape == (2, 5)
    true = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    pred = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    assert get_l2_loss(true, pred, True).shape == (1, 2, 5)
    assert get_l2_loss(true, pred, False).shape == (1, 2, 5)


def test_get_image_loss():
    true = torch.tensor([1, 2, 3, 4, 5])
    pred = torch.tensor([1, 2, 3, 4, 5])
    assert get_image_loss(true, pred, True).shape == (1, 1, 5)
    assert get_image_loss(true, pred, False).shape == (1, 5)
    true = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    pred = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert get_image_loss(true, pred, True).shape == (2, 1, 5)
    assert get_image_loss(true, pred, False).shape == (2, 5)
    true = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    pred = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    assert get_image_loss(true, pred, True).shape == (1, 2, 5)
    assert get_image_loss(true, pred, False).shape == (1, 2, 5)
