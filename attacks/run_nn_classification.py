import numpy as np
import pandas as pd
import os
import json
import multiprocessing

from torch.utils.data import DataLoader, Dataset, Subset
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from attacks_config import NAMES as attacks

from sklearn.metrics import roc_curve

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

attacks.pop()  # remove carlini
attacks.pop()  # remove black-box

path = "laion"
device = "cuda:0"
SAMPLES_TO_TAKE = 5_000
EXPERIMENTS_CNT = 100
SAMPLES_PER_EXPERIMENT = 500
EPOCHS = 10
loss_names = ["noise_loss", "latent_loss", "image_loss"]
get_attack_filename = lambda is_member, attack: os.path.join(
    "data/mia_out/laion",
    path,
    f"__{'members' if is_member else 'nonmembers'}-{attack}.npz",
)
crop_data = lambda data: data[:SAMPLES_TO_TAKE, :, :]
get_averaged_loss_data = lambda data: np.mean(data, axis=-1)
get_tpr_at_fpr_1_from_tpr_fpr = lambda tpr, fpr: tpr[np.sum(fpr <= 0.01) - 1]


def get_loss_thrs_preds(X, X_max=None, members_lower=True):
    if X_max is None:
        X_max = X.max()
    if members_lower:
        return 1 - (X / X_max)
    else:
        return X / X_max


def get_data_for_experiment_gen(m_array, nm_array):
    np.random.seed(123)
    for _ in range(EXPERIMENTS_CNT):
        indices = np.random.permutation(SAMPLES_TO_TAKE)
        train_data = np.concatenate(
            [
                m_array[indices[SAMPLES_PER_EXPERIMENT:]],
                nm_array[indices[SAMPLES_PER_EXPERIMENT:]],
            ],
            axis=0,
        )
        test_data = np.concatenate(
            [
                m_array[indices[:SAMPLES_PER_EXPERIMENT]],
                nm_array[indices[:SAMPLES_PER_EXPERIMENT]],
            ],
            axis=0,
        )
        train_labels = np.concatenate(
            [
                np.ones(SAMPLES_TO_TAKE - SAMPLES_PER_EXPERIMENT),
                np.zeros(SAMPLES_TO_TAKE - SAMPLES_PER_EXPERIMENT),
            ],
            axis=0,
        )
        test_labels = np.concatenate(
            [
                np.ones(SAMPLES_PER_EXPERIMENT),
                np.zeros(SAMPLES_PER_EXPERIMENT),
            ],
            axis=0,
        )
        yield train_data, test_data, train_labels, test_labels


def get_init_data(attack):
    members, nonmembers = [], []
    for loss_name in loss_names:
        members.append(
            get_averaged_loss_data(
                crop_data(np.load(get_attack_filename(True, attack))[loss_name])
            )
        )
        nonmembers.append(
            get_averaged_loss_data(
                crop_data(np.load(get_attack_filename(False, attack))[loss_name])
            )
        )
    members = np.concatenate(members, axis=-1)
    nonmembers = np.concatenate(nonmembers, axis=-1)
    return members, nonmembers


def get_net(input_size, hidden_size=10):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid(),
    )


class MembershipDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ExperimentProcess(multiprocessing.Process):
    def __init__(self, attack, results_q):
        super().__init__()
        self.attack = attack
        self.results_q = results_q

    def run(self):
        members, nonmembers = get_init_data(self.attack)
        results = []
        for train_data, test_data, train_labels, test_labels in tqdm(
            get_data_for_experiment_gen(members, nonmembers),
            desc=f"Attack: {self.attack}",
            total=EXPERIMENTS_CNT,
        ):
            results.append(
                self.run_experiment(train_data, test_data, train_labels, test_labels)
            )
        self.results_q.put((self.attack, results))

    def run_experiment(self, train_data, test_data, train_labels, test_labels):
        model = get_net(train_data.shape[1]).to(device)
        train_dataset = MembershipDataset(train_data, train_labels)
        train_dataset, eval_dataset = self.split_train_to_train_and_eval(train_dataset)
        test_dataset = MembershipDataset(test_data, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.train(model, train_loader, eval_loader)  # train model

        preds = self.test(model, test_loader)  # test model
        fpr, tpr, _ = roc_curve(test_labels, preds)
        return get_tpr_at_fpr_1_from_tpr_fpr(tpr, fpr)

    def split_train_to_train_and_eval(self, dataset):
        indices = np.random.permutation(len(dataset))
        train_indices = indices[: int(0.8 * len(indices))]
        eval_indices = indices[int(0.8 * len(indices)) :]
        train_dataset = Subset(dataset, train_indices)
        eval_dataset = Subset(dataset, eval_indices)
        return train_dataset, eval_dataset

    def early_stopping(self, model, eval_loader, criterion, best_eval_loss, best_model):
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for data, target in eval_loader:
                output = model(data.to(device)).squeeze()
                eval_loss += criterion(output, target.to(device)).item()
        eval_loss /= len(eval_loader.dataset)
        if eval_loss < best_eval_loss:
            return eval_loss, model, False
        else:
            return best_eval_loss, best_model, True

    def train(self, model, train_loader, eval_loader):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        best_eval_loss = np.inf
        best_model = None
        for epoch in range(100):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data.to(device)).squeeze()
                loss = criterion(output, target.to(device))
                loss.backward()
                optimizer.step()
            best_eval_loss, best_model, stop = self.early_stopping(
                model, eval_loader, criterion, best_eval_loss, best_model
            )
            if stop:
                break

        return best_model

    def test(self, model, test_loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for data, _ in test_loader:
                output = model(data.to(device))
                preds.append(output.cpu())
        return torch.cat(preds).numpy()


def main():
    results = {}
    results_q = multiprocessing.Queue()
    for attack in attacks:
        ExperimentProcess(attack, results_q).start()
    for _ in tqdm(range(len(attacks)), desc="Waiting for results"):
        attack, result = results_q.get()
        results[attack] = result
    with open("nn_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
