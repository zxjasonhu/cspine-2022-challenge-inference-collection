import numpy as np
import pandas as pd

from utils import create_double_cv


def get_percentiles(x):
    return np.percentile(x, [0, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 100])


df = pd.read_csv("../../data/train_vertebra_chunks_with_sizes.csv")
print(get_percentiles(df.z))
print(get_percentiles(df.h))
print(get_percentiles(df.w))
train_df = pd.read_csv("../../data/train.csv")
df["StudyInstanceUID"] = df.filename.apply(lambda x: x.split("_")[0])
df["level"] = df.filename.apply(lambda x: x.split("_")[1].replace(".npy", ""))
df = df.merge(train_df, on="StudyInstanceUID")
df["fracture"] = 0

for level in [f"C{i}" for i in range(1, 8)]:
    df.loc[df.level == level, "fracture"] = df.loc[df.level == level, level]

df = create_double_cv(df, "StudyInstanceUID", 5, 5)
df.to_csv("../../data/train_vertebra_chunks_kfold.csv", index=False)
