import glob
import numpy as np
import pandas as pd
import pickle
import torch


def load_pickle(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)


predictions = glob.glob("../../predictions/pre000/fold*pkl")
df_list = []
for pred in predictions:
    pred = load_pickle(pred)
    probas = torch.cat([pred[i][0] for i in range(len(pred))])
    images = np.concatenate([pred[i][1] for i in range(len(pred))])
    probas = torch.sigmoid(probas).numpy()
    tmp_df = pd.DataFrame({"imgfile": images, "p": probas})
    tmp_df = tmp_df.sort_values("imgfile", ascending=True)
    df_list.append(tmp_df)

df = df_list[0].copy()
for each_df in df_list[1:]:
    df["p"] += each_df["p"]

df["p"] /= 5.
df["imgfile"] = df.imgfile.apply(lambda x: x.split(",")[1].replace("../data/pngs/", ""))
df["StudyInstanceUID"] = df.imgfile.apply(lambda x: x.split("/")[0])
df["slice_number"] = df.imgfile.apply(lambda x: x.split("/")[-1].replace(".png", ""))
df["study_slice"] = df.StudyInstanceUID + "_" + df.slice_number.astype("str")

level_df = pd.read_csv("../../data/train_cspine_levels.csv")
level_df["study_slice"] = level_df.StudyInstanceUID + "_" + level_df.slice_number.astype("str")
train_df = pd.read_csv("../../data/train.csv")

df = df.merge(level_df, on="study_slice")

train_df = train_df[train_df.StudyInstanceUID.isin(df.StudyInstanceUID.tolist())]

false_negatives = []
for study_id, study_df in train_df.groupby("StudyInstanceUID"):
    tmp_df = tmp_df[tmp_df.StudyInstanceUID == study_id]
    for level in [f"C{i+1}" for i in range(7)]:
        if study_df[level] == 1:
            # Set slices that do NOT contain that level to 0
            tmp_df.loc[tmp_df[level] != 1, "p"] = 0
            # Set slices that DO contain that level to 1 IF predicted probability is >= 0.5
            tmp_df.loc[tmp_df[level] == 1, "p"] = (tmp_df.loc[tmp_df[level] == 0, "p"] >= 0.5).astype("float")
            # If no slices at that level were predicted positive, then store the study-level for
            # manual review
            if tmp_df.loc[tmp_df[level] == 1, "p"].sum() == 0:
                false_negatives.append(f"{study_id}_{level}")



