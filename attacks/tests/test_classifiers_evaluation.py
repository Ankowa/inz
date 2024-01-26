from attacks.classifiers_evaluation import (
    get_loss_thrs_preds,
    get_data_for_experiment_gen,
)
import torch
import numpy as np


def test_get_loss_thrs_preds():
    X = torch.tensor([1, 2, 3, 4, 5])
    X_max = 5
    assert torch.allclose(
        get_loss_thrs_preds(X, X_max, True),
        torch.tensor([0.8, 0.6, 0.4, 0.2, 0]),
    )
    assert torch.allclose(
        get_loss_thrs_preds(X, X_max, False),
        torch.tensor([0.2, 0.4, 0.6, 0.8, 1]),
    )
    X = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    X_max = 5
    assert torch.allclose(
        get_loss_thrs_preds(X, X_max, True),
        torch.tensor(
            [
                [0.8, 0.6, 0.4, 0.2, 0],
                [0.8, 0.6, 0.4, 0.2, 0],
            ]
        ),
    )
    assert torch.allclose(
        get_loss_thrs_preds(X, X_max, False),
        torch.tensor(
            [
                [0.2, 0.4, 0.6, 0.8, 1],
                [0.2, 0.4, 0.6, 0.8, 1],
            ]
        ),
    )
    X = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    X_max = 5
    assert torch.allclose(
        get_loss_thrs_preds(X, X_max, True),
        torch.tensor(
            [
                [[0.8, 0.6, 0.4, 0.2, 0], [0.8, 0.6, 0.4, 0.2, 0]],
            ]
        ),
    )

    assert torch.allclose(
        get_loss_thrs_preds(X, X_max, False),
        torch.tensor(
            [
                [[0.2, 0.4, 0.6, 0.8, 1], [0.2, 0.4, 0.6, 0.8, 1]],
            ]
        ),
    )


def test_get_data_for_experiment_gen():
    gen = get_data_for_experiment_gen(
        np.array([[1, 2, 3, 4, 5]]),
        np.array([[1, 2, 3, 4, 5]]),
    )
    for _ in range(10):
        train_data, test_data, train_labels, test_labels = next(gen)
        assert train_data.shape == (8, 5)
        assert test_data.shape == (2, 5)
        assert train_labels.shape == (8,)
        assert test_labels.shape == (2,)
        assert np.all(train_data == test_data)
        assert np.all(train_labels == test_labels)
        assert np.all(train_data == np.array([[1, 2, 3, 4, 5]]))
        assert np.all(train_labels == np.array([1, 1, 1, 1, 0, 0, 0, 0]))
        assert np.all(test_data == np.array([[1, 2, 3, 4, 5]]))
        assert np.all(test_labels == np.array([1, 1]))
