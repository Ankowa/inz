import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import entropy, wasserstein_distance
import torchvision

from sklearn.metrics import roc_curve
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import auc, RocCurveDisplay

from attacks_config import NAMES as attacks
from attacks_config import LABELS as attack_labels
from utils import setup_matplotlib
from attack_utils import get_batch_generator

setup_matplotlib()

SAMPLES_TO_TAKE = 5_000
loss_names = ["noise_loss", "latent_loss", "image_loss"]
loss_labels = ["Model Loss", "Latent Error", "Pixel Error"]
path = "data/mia_out/laion"

attacks.pop(-2)  # remove black-box
attack_labels.pop(-2)  # remove black-box
attack_map = dict(
    zip([f"Attack {idx}" for idx in range(1, len(attacks) + 1)], attack_labels)
)

get_attack_filename = lambda is_member, attack: os.path.join(
    path, f"__{'members' if is_member else 'nonmembers'}-{attack}.npz"
)

crop_data = lambda data: data[:SAMPLES_TO_TAKE, :, :]
get_averaged_loss_data = lambda data: np.mean(data, axis=-1)
get_tpr_at_fpr_1_from_tpr_fpr = lambda tpr, fpr: tpr[np.sum(fpr <= 0.01) - 1]

get_array = lambda x: np.array(
    eval(" ".join(x.split()).replace("\n", "").replace(" ", ", "))
)
get_mean = lambda x: np.mean(get_array(x)) * 100
get_std = lambda x: np.std(get_array(x)) * 100
get_latex_entry = lambda x: f"{get_mean(x):.2f}\%$\pm${get_std(x):.2f}"


def get_loss_thrs_preds(X, X_max=None, members_lower=True):
    if X_max is None:
        X_max = X.max()
    if members_lower:
        return 1 - (X / X_max)
    else:
        return X / X_max


def calc_loss_based_attack_performance(m_data, nm_data):
    samples_per_test = 1_000
    tests_cnt = 1_00
    results = []
    np.random.seed(123)

    for _ in range(tests_cnt):
        tmp_m = m_data[
            np.random.choice(m_data.shape[0], samples_per_test, replace=False)
        ]
        tmp_nm = nm_data[
            np.random.choice(nm_data.shape[0], samples_per_test, replace=False)
        ]
        tmp_data = np.concatenate([tmp_m, tmp_nm], axis=0)
        tmp_labels = np.concatenate(
            [np.ones(samples_per_test), np.zeros(samples_per_test)], axis=0
        )
        test_results = []
        for idx in range(tmp_data.shape[1]):
            preds = get_loss_thrs_preds(tmp_data[:, idx], members_lower=True)
            fpr, tpr, _ = roc_curve(tmp_labels, preds)
            test_results.append(get_tpr_at_fpr_1_from_tpr_fpr(tpr, fpr))
        results.append(np.array(test_results).reshape(1, -1))
    results = np.concatenate(results, axis=0)
    return results


def get_entropies():
    entropies = []
    for attack, attack_label in tqdm(zip(attacks, attack_labels)):
        members = np.load(get_attack_filename(True, attack), allow_pickle=True)
        nonmembers = np.load(get_attack_filename(False, attack), allow_pickle=True)
        dfs = []
        for loss_name in loss_names:
            tmp_entropies = []
            m_data = get_averaged_loss_data(crop_data(members[loss_name]))
            nm_data = get_averaged_loss_data(crop_data(nonmembers[loss_name]))
            for y_idx in range(m_data.shape[1]):
                tmp_entropies.append(entropy(m_data[y_idx], nm_data[y_idx]))
            tmp_df = pd.DataFrame(tmp_entropies, columns=["value"])
            tmp_df["loss"] = loss_name
            performance = calc_loss_based_attack_performance(m_data, nm_data)
            tmp_df["attack_performance_mean"] = performance.mean(axis=0) * 100
            tmp_df["attack_performance_max"] = performance.max(axis=0) * 100
            tmp_df["attack_performance_min"] = performance.min(axis=0) * 100
            tmp_df["attack_performance_std"] = performance.std(axis=0) * 100

            dfs.append(tmp_df)
        df = pd.concat(dfs, axis=0)
        df["attack"] = attack_label
        entropies.append(df)
    return pd.concat(entropies)


def get_bootstrapping_plot(entropies):
    _, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=True)
    for _, (loss_name, loss_label, ax) in enumerate(zip(loss_names, loss_labels, axs)):
        for metric in [
            "attack_performance_mean",
            "attack_performance_max",
            "attack_performance_min",
        ]:
            sns.lineplot(
                x="attack",
                y=metric,
                data=entropies[entropies["loss"] == loss_name],
                ax=ax,
            )
        ax.set_title(loss_label)
        ax.set_ylabel("TPR at FPR=1%")
        ax.set_xlabel("")
        ax.xaxis.axes.set_xticks([])
    plt.figlegend(
        ["Mean attack performance", "Max attack performance", "Min attack performance"],
        loc="lower center",
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    plt.tight_layout()
    plt.savefig("plots/randomization_importance.pdf", dpi=300)
    plt.show()


def get_attacks_latex_table(entropies):
    data = entropies.copy()
    data["attack_performance_mean_std"] = (
        data["attack_performance_mean"].apply(lambda x: round(x, 2)).astype(str)
        + "\%$\pm$"
        + data["attack_performance_std"].apply(lambda x: round(x, 2)).astype(str)
    )
    data = (
        data.groupby(["loss", "attack"])["attack_performance_mean_std"]
        .max()
        .reset_index()
    )
    data.rename(columns={"attack": "Attack", "loss": "Loss"}, inplace=True)
    data.Loss = data.Loss.map(
        {
            loss_name: loss_label
            for loss_name, loss_label in zip(loss_names, loss_labels)
        }
    )
    data = data.pivot(
        index="Loss", columns="Attack", values="attack_performance_mean_std"
    )
    data = data[attack_labels].T
    data = data[loss_labels]
    data.index.name = "Setting"
    data.to_latex(
        "plots/attack_performance.tex", float_format="%.2f", index=True, escape=False
    )


def get_attack_data_labels(m_data, nm_data, loss_name):
    data = np.concatenate(
        [
            get_averaged_loss_data(m_data[loss_name]),
            get_averaged_loss_data(nm_data[loss_name]),
        ],
        axis=0,
    )
    labels = np.concatenate(
        [np.ones(len(m_data[loss_name])), np.zeros(len(nm_data[loss_name]))]
    )
    return data, labels


def get_best_fpr_tpr(m_data, nm_data, loss_names):
    best_result = -np.inf
    best_tpr = None
    best_fpr = None
    for loss_name in loss_names:
        try:
            data, labels = get_attack_data_labels(m_data, nm_data, loss_name)
        except:
            continue
        for idx in range(data.shape[1]):
            preds = get_loss_thrs_preds(data[:, idx], members_lower=True)
            fpr, tpr, _ = roc_curve(labels, preds)
            result = get_tpr_at_fpr_1_from_tpr_fpr(tpr, fpr)
            if result > best_result:
                best_result = result
                best_tpr = tpr
                best_fpr = fpr
    return best_fpr, best_tpr


def get_teaser():
    setting_names = ["Baseline loss threshold", "Partial denoising", "Reversed noising"]
    pokemon_colors = ["orange", "red", "firebrick"]
    laion_colors = ["blue", "lime", "darkgreen"]
    dataset_names = ["POKEMON", "LAION-mi"]
    pokemon_members_carlini = np.load(
        os.path.join("pokemons", "members_final.npz"), allow_pickle=True
    )
    pokemon_members_1 = np.load(
        os.path.join("pokemons", "__members-embeddings-setting-1.npz"),
        allow_pickle=True,
    )
    pokemon_members_16 = np.load(
        os.path.join("pokemons", "__members-embeddings-setting-16.npz"),
        allow_pickle=True,
    )
    pokemon_nonmembers_carlini = np.load(
        os.path.join("pokemons", "nonmembers_final.npz"), allow_pickle=True
    )
    pokemon_nonmembers_1 = np.load(
        os.path.join("pokemons", "__nonmembers-embeddings-setting-1.npz"),
        allow_pickle=True,
    )
    pokemon_nonmembers_16 = np.load(
        os.path.join("pokemons", "__nonmembers-embeddings-setting-16.npz"),
        allow_pickle=True,
    )
    laion_members_carlini = np.load(
        get_attack_filename(True, "embeddings-carlini"), allow_pickle=True
    )
    laion_members_1 = np.load(get_attack_filename(True, attacks[0]), allow_pickle=True)
    laion_members_16 = np.load(
        get_attack_filename(True, attacks[15]), allow_pickle=True
    )
    laion_nonmembers_carlini = np.load(
        get_attack_filename(False, "embeddings-carlini"), allow_pickle=True
    )
    laion_nonmembers_1 = np.load(
        get_attack_filename(False, attacks[0]), allow_pickle=True
    )
    laion_nonmembers_16 = np.load(
        get_attack_filename(False, attacks[15]), allow_pickle=True
    )
    plt.figure(figsize=(8, 5))
    for setting_name, color, dataset_name, m_data, nm_data in zip(
        setting_names * 2,
        pokemon_colors + laion_colors,
        [dataset_names[0]] * 3 + [dataset_names[1]] * 3,
        [
            pokemon_members_carlini,
            pokemon_members_1,
            pokemon_members_16,
            laion_members_carlini,
            laion_members_1,
            laion_members_16,
        ],
        [
            pokemon_nonmembers_carlini,
            pokemon_nonmembers_1,
            pokemon_nonmembers_16,
            laion_nonmembers_carlini,
            laion_nonmembers_1,
            laion_nonmembers_16,
        ],
    ):
        fpr, tpr = get_best_fpr_tpr(m_data, nm_data, loss_names)
        plt.plot(fpr, tpr, color=color, label=f"{dataset_name}: {setting_name}")
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(0.001, 1)
    plt.ylim(0.001, 1)
    plt.plot((0.001, 1), (0.001, 1), color="black", linestyle="dashed")
    plt.legend()
    plt.title("ROC curve")
    plt.savefig("plots/teaser.pdf")
    plt.show()


def get_clf_attacks_latex_table():
    results = pd.read_csv("laion/clf_results_no_svc_grid_search.csv", index_col=0)
    with open("nn_results.json", "r") as f:
        nn_results = json.load(f)
    nn_results_df = pd.DataFrame(
        pd.DataFrame.from_dict(nn_results, orient="index")
        .apply(np.array, axis=1)
        .astype(str),
        columns=["neural_net"],
    ).T
    results = pd.concat([results, nn_results_df])

    settings = results.columns
    models = results.index
    model_labels_short = ["LR", "DTC", "RFC", "NN"]
    df = results.applymap(get_latex_entry)
    df.rename(
        columns={
            setting: attack_label
            for setting, attack_label in zip(settings, attack_labels)
        },
        inplace=True,
    )
    df = df.rename(
        index={
            model: model_label for model, model_label in zip(models, model_labels_short)
        }
    )
    df = df.T
    df.index.name = "Setting"
    df.columns.name = "Classifier class"
    df.to_latex(
        "plots/laion_clf_results.tex", float_format="%.2f", index=True, escape=False
    )
    return results, models, settings


def get_clf_bootstrapping_plot(results, models, settings):
    data = []
    for _, row in results.iterrows():
        row_data = []
        for col in results.columns:
            row_data.append(row.apply(get_array)[col])
        data.append(
            np.concatenate(np.array(row_data).reshape(1, -1), axis=0).reshape(
                -1, 16, 100
            )
        )
    data = np.concatenate(data, axis=0)

    dfs = []
    for z_idx, model in enumerate(models):
        m_dfs = []
        for y_idx, setting in enumerate(settings):
            df = pd.DataFrame(
                {
                    "results_mean": [data[z_idx, y_idx, :].flatten().mean()],
                    "results_min": [data[z_idx, y_idx, :].flatten().min()],
                    "results_max": [data[z_idx, y_idx, :].flatten().max()],
                    "setting": [setting],
                    "model": [model],
                }
            )
            m_dfs.append(df)
        dfs.append(pd.concat(m_dfs, axis=0))
    df = pd.concat(dfs, axis=0)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_laion_mi_vis(s):
    class DummyArgs:
        def __init__(self, input_dir, batch_size):
            self.input_dir = input_dir
            self.batch_size = batch_size

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(512),
            torchvision.transforms.CenterCrop(512),
        ]
    )
    args = DummyArgs(s, 16)
    generator = get_batch_generator(
        args,
        transform,
        generations_per_sample=1,
        is_parquet=True,
        shuffle=True,
    )
    images, _ = next(generator)
    images = images[range(0, 80, 5)]
    images = [torchvision.transforms.ToPILImage()(image) for image in images]
    grid = image_grid(images, 4, 4)
    grid.save(f"plots/laion_mi_vis_{s}.png")


def get_pokemon_roc():
    path = "pokemons"
    loss_name = "noise_loss"
    checkpoints = [
        "0",
        "5000",
        "10000",
        "15000",
        "20000",
        "25000",
        "final",
    ]
    names = [
        "SD-v.1.4",
        "5000",
        "10000",
        "15000",
        "20000",
        "25000",
        "30000",
    ]
    load_data = (
        lambda is_member, attack: np.load(
            os.path.join(
                path, f"{'members' if is_member else 'nonmembers'}_{attack}.npz"
            ),
            allow_pickle=True,
        )[loss_name]
        .reshape(-1, 5)
        .mean(axis=-1)
    )
    results = []
    _, ax = plt.subplots(1, 1, figsize=(7, 7))
    for checkpoint in checkpoints:
        members = load_data(True, checkpoint)
        nonmembers = load_data(False, checkpoint)
        data = np.concatenate([members, nonmembers])
        labels = np.concatenate([np.ones_like(members), np.zeros_like(nonmembers)])
        preds = get_loss_thrs_preds(data, members_lower=True)
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="image distance"
        )
        display.plot(ax=ax)
        results.append(get_tpr_at_fpr_1_from_tpr_fpr(tpr, fpr))

    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([1e-3, 1.0])
    ax.set_ylim([1e-3, 1.0])
    ax.legend(
        [
            f"{name}, TPR@FPR=1%: {result*100:.1f}%"
            for name, result in zip(names, results)
        ]
        + ["random"],
        loc="lower right",
    )
    plt.title("Pokemon ROC curves for different checkpoints")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("plots/pokemons_roc.pdf", dpi=300, bbox_inches="tight")
    return results, names, load_data, checkpoints


def get_pokemon_ws_dist(results, names, load_data, checkpoints):
    results = []
    _, ax = plt.subplots(1, 1, figsize=(7, 7))

    for checkpoint in checkpoints:
        members = load_data(True, checkpoint)
        nonmembers = load_data(False, checkpoint)
        results.append(wasserstein_distance(members, nonmembers))

    ax.bar(names, results, color="tab:blue", edgecolor="black")
    ax.bar_label(ax.containers[0], fmt="%.2f", label_type="edge", fontsize=12)
    ax.set_xticklabels(names, rotation=45)
    plt.title("Pokemons members vs non-members noise loss")
    plt.xlabel("Checkpoint")
    plt.ylabel("Wasserstein Distance")
    plt.savefig("plots/pokemons_wasserstein.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    entropies = get_entropies()
    get_bootstrapping_plot(entropies)
    get_attacks_latex_table(entropies)
    get_teaser()
    results, models, settings = get_clf_attacks_latex_table()
    get_clf_bootstrapping_plot(results, models, settings)
    for s in ["members", "nonmembers"]:
        get_laion_mi_vis(s)
    results, names, load_data, checkpoints = get_pokemon_roc()
    get_pokemon_ws_dist(results, names, load_data, checkpoints)


if __name__ == "__main__":
    main()
