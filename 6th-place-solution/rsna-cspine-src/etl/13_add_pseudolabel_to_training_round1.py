import numpy as np
import os.path as osp
import pandas as pd
import pickle
import torch


def load_pickle(fp):
    with open(fp, "rb") as f:
        preds = pickle.load(f)
    probas = torch.sigmoid(torch.cat([p[0] for p in preds])).numpy()
    filenames = np.concatenate([p[1] for p in preds])
    return probas, filenames


def sharpen(x, T):
    x = np.asarray(x)
    x = x ** (1 / T)
    return x / np.sum(x)


df = pd.read_csv("../../data/train_initial_for_2d_pseudolabel_with_split.csv")
pickle_paths = [osp.join("../../predictions/pre000/", pkl_path + ".pkl") for pkl_path in ["fold0_seed870", "fold0_seed880", "fold0_seed890"]]
preds = [load_pickle(pkl_path) for pkl_path in pickle_paths]
for p in preds:
    assert np.mean(preds[0][1] == p[1]) == 1

probas = np.mean([p[0] for p in preds], axis=0)
selected = (probas >= 0.7) | (probas <= 0.3)
probas = probas[selected]
filenames = preds[0][1][selected]
dist = np.stack([probas, 1 - probas], axis=1)
sharpened = np.stack([sharpen(dist[i], 0.5) for i in range(len(dist))])
filenames = [f.replace("../data/pngs/", "") for f in filenames]
add_df = pd.DataFrame({"filename": filenames, "fracture": np.round(sharpened[:, 0]), "outer": -1})
print(add_df.fracture.value_counts())
assert len(list(set(add_df.filename) & set(df.filename))) == 0
df = pd.concat([df[["filename", "fracture", "outer"]], add_df])
df.to_csv("../../data/train_round1_for_2d_pseudolabel_with_split.csv", index=False)
print(np.mean(df.fracture >= 0.5))