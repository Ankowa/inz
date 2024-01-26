import numpy as np
import pandas as pd
import os
import multiprocessing

from tqdm import tqdm
from attacks_config import NAMES as attacks

from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

attacks.pop()  # remove carlini
attacks.pop()  # remove black-box

models = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}

models_grid_search_params = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10, 100],
    },
    "decision_tree": {
        "max_depth": [2, 8, 16],
        "min_samples_split": [2, 8, 16],
    },
    "random_forest": {
        "n_estimators": [10, 100, 1000],
        "max_depth": [2, 8, 16],
    },
}
path = "LAION"
SAMPLES_TO_TAKE = 5_000
EXPERIMENTS_CNT = 100
SAMPLES_PER_EXPERIMENT = 500
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


class ExperimentProcess(multiprocessing.Process):
    def __init__(
        self, train_data, test_data, train_labels, test_labels, model_name, queue
    ):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.model_name = model_name
        self.queue = queue

    def run(self):
        params = self.perform_grid_search()
        model = models[self.model_name](**params)
        model.fit(self.train_data, self.train_labels)
        preds = model.predict_proba(self.test_data)[:, 1]
        fpr, tpr, _ = roc_curve(self.test_labels, preds)
        self.queue.put(get_tpr_at_fpr_1_from_tpr_fpr(tpr, fpr))

    def perform_grid_search(self):
        model = models[self.model_name]()
        params = models_grid_search_params[self.model_name]
        grid_search = GridSearchCV(
            model,
            params,
            scoring="roc_auc",
            n_jobs=1,
            verbose=0,
            cv=5,
        )
        grid_search.fit(self.train_data, self.train_labels)
        return grid_search.best_params_


def single_experiment(attack, model_name):
    members, nonmembers = get_init_data(attack)
    results_q = multiprocessing.Queue()
    results = []
    processes = []
    for train_data, test_data, train_labels, test_labels in get_data_for_experiment_gen(
        members, nonmembers
    ):
        p = ExperimentProcess(
            train_data, test_data, train_labels, test_labels, model_name, results_q
        )
        p.start()
        processes.append(p)
    for _ in tqdm(
        range(EXPERIMENTS_CNT), desc=f"Model: {model_name}, Attack: {attack}"
    ):
        results.append(results_q.get())
    for p in processes:
        p.join()
    return np.array(results)


def single_attack(attack):
    results = {}
    for model_name in models.keys():
        results[model_name] = single_experiment(attack, model_name)
    return results


def main():
    results = {}
    for attack in tqdm(attacks, desc="Main run"):
        results[attack] = single_attack(attack)
    df = pd.DataFrame.from_dict(results)
    df.to_csv("data/results/laion/clf_results_no_svc_grid_search.csv")


if __name__ == "__main__":
    main()
