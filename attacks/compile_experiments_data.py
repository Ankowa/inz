import numpy as np
import os
from attacks_config import NAMES


def save_merged(data_list, filename):
    data_fin = dict()
    for batch in data_list:
        for key in batch:
            if key == "prompts":
                if key not in data_fin:
                    data_fin[key] = []
                data_fin[key].extend(batch[key])
            else:
                if key not in data_fin:
                    data_fin[key] = batch[key]
                    continue
                data_fin[key] = np.concatenate([data_fin[key], batch[key]], axis=0)
    np.savez(f"LAION/{filename}", **data_fin)


def run(attack, dataset):
    embeddings_members, embeddings_nonmembers = set(), set()
    members, nonmembers = [], []
    for file in sorted(os.listdir(f"data/mia_out/{dataset}")):
        for s, emb_s, name in zip(
            [members, nonmembers],
            [embeddings_members, embeddings_nonmembers],
            ["members", "nonmembers"],
        ):
            if file != f"__{name}-{attack}.npz":
                continue
            data = np.load("LAION/" + file, allow_pickle=True)
            emb_s.update(data["prompts"])
            s.append(data)

    save_merged(members, f"members-{attack}")
    save_merged(nonmembers, f"nonmembers-{attack}")


def main():
    for attack in NAMES:
        for dataset in ["laion", "pokemon"]:
            run(attack, dataset)


if __name__ == "__main__":
    main()
