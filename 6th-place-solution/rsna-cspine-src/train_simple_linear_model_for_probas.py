import glob
import numpy as np
import os.path as osp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def get_train_valid_subsets(probas, folds_dict, fold):
    train = np.vstack([v for k, v in probas.items() if k not in folds_dict[fold]])
    test = np.vstack([v for k, v in probas.items() if k in folds_dict[fold]])
    return train, test


def competition_metric(p, t):
    # p.shape = t.shape = (N, 8)
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p).float()
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float()
    loss_matrix = F.binary_cross_entropy(p, t, reduction="none")
    # loss_matrix.shape = (N, 8)
    columnwise_losses = []
    for col in range(loss_matrix.shape[1]):
        weights = t[:, col] + 1 # positives are weighted 2x
        columnwise_losses.append(((loss_matrix[:, col] * weights).sum() / weights.sum()).item())
    columnwise_losses[-1] *= 7.0
    return np.sum(columnwise_losses) / 14.0


def train(X_train, X_test, y_train, y_test, model, loss_fn, eval_metric, optimizer, num_iterations):
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    for it in tqdm(range(num_iterations)):
        model.train()
        optimizer.zero_grad()
        p = model(X_train)
        loss = loss_fn(p, y_train)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            p_valid = torch.sigmoid(model(X_test))
        eval_value = eval_metric(p_valid, y_test)
        print(f"EVALUATION METRIC : {eval_value.item(): 0.3f}")
    return model


class SimpleLinear(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7, 14)
        self.layer2 = nn.Linear(14, 8)

    def forward(self, x):
        return self.layer2(self.layer1(x))


PROBAS_DIR = "../data/train-chunk000-probas/"
probas_files = glob.glob(osp.join(PROBAS_DIR, "*.npy"))

study_labels_df = pd.read_csv("../data/train.csv")

folds_df = pd.read_csv("../data/train_kfold.csv")
folds_dict = {}
for outer_fold, oof_df in folds_df.groupby("outer"):
    folds_dict[outer_fold] = oof_df.StudyInstanceUID.tolist()

probas = {osp.basename(file_name).replace(".npy", ""): np.load(file_name) for file_name in probas_files}
study_labels_df = study_labels_df[study_labels_df.StudyInstanceUID.isin([*probas])]
labels = {study_id: study_df[[f"C{i+1}" for i in range(7)] + ["patient_overall"]].values
          for study_id, study_df in study_labels_df.groupby("StudyInstanceUID")}

X_train_0, X_test_0 = get_train_valid_subsets(probas, folds_dict, 0)
y_train_0, y_test_0 = get_train_valid_subsets(labels, folds_dict, 0)
model = SimpleLinear()
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
loss_fn = nn.BCEWithLogitsLoss()
train(X_train_0, X_test_0, y_train_0, y_test_0, model, loss_fn, competition_metric, optimizer, num_iterations=1000000)




import torch, numpy as np

from skp.models.sequence import DualTransformerV2


X = torch.from_numpy(np.ones((8, 128, 512))).float()
mask = torch.from_numpy(np.ones((8, 128)))
model = DualTransformerV2(1, 8)
model((X, mask))


