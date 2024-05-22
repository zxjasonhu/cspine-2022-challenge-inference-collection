import numpy as np
import torch
from torch.utils.data import Dataset

pred_cols = [
    "pred_frac_c1",
    "pred_frac_c2",
    "pred_frac_c3",
    "pred_frac_c4",
    "pred_frac_c5",
    "pred_frac_c6",
    "pred_frac_c7",
]
label_cols = [
    "label_frac_c1",
    "label_frac_c2",
    "label_frac_c3",
    "label_frac_c4",
    "label_frac_c5",
    "label_frac_c6",
    "label_frac_c7",
]


class RSNAStackerDataset(Dataset):

    def __init__(self, df, mode):
        self.df = df.copy().reset_index(drop=True)
        self.mode = mode

        self.feature_cols = []
        self.label_cols = label_cols.copy()

        df = self.df

        features = []

        for j, l in enumerate(pred_cols):
            features.append(df.groupby("StudyInstanceUID")[l].mean().values.reshape(-1, 1))
            features.append(df.groupby("StudyInstanceUID")[l].min().values.reshape(-1, 1))
            features.append(df.groupby("StudyInstanceUID")[l].max().values.reshape(-1, 1))

        features.append(df.groupby("StudyInstanceUID").size().values.reshape(-1, 1) / 1_000)

        labels = (
            df.groupby("StudyInstanceUID")[
                self.label_cols
            ]
            .max()
            .reset_index(drop=True)
        )
        labels["label_overall"] = labels[
            self.label_cols
        ].max(axis=1)

        self.X = np.concatenate(features, axis=1)
        self.y = labels.values

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        X = self.X[idx]
        y = self.y[idx]

        return torch.FloatTensor(X), torch.FloatTensor(y)

    def __len__(self):
        return self.df.StudyInstanceUID.nunique()