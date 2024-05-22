import pandas as pd


folds_df = pd.read_csv("../../data/train_kfold.csv")
df = pd.read_csv("../../data/train.csv")
df = df.merge(folds_df, on="StudyInstanceUID")
df["filename"] = df.StudyInstanceUID + ".npy"
df.to_csv("../../data/train_seq_kfold.csv", index=False)
