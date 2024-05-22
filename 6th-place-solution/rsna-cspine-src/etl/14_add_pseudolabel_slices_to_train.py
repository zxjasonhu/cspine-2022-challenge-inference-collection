import glob
import numpy as np
import pandas as pd
import pickle
import torch


def load_pickle(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)


predictions = np.sort(glob.glob("../../predictions/pre000/fold*pkl"))
df_list = []
for pred in predictions:
    print(f"Loading {pred} ...")
    pred = load_pickle(pred)
    probas = torch.cat([pred[i][0] for i in range(len(pred))])
    images = np.concatenate([pred[i][1] for i in range(len(pred))])
    probas = torch.sigmoid(probas).numpy()
    tmp_df = pd.DataFrame({"filename_2dc": images, "fracture": probas})
    tmp_df = tmp_df.sort_values("filename_2dc", ascending=True)
    df_list.append(tmp_df)

df = df_list[0].copy()
# for each_df in df_list[1:]:
#     df["fracture"] += each_df["fracture"]
#
# df["fracture"] /= 5.
df["filename_2dc"] = df.filename_2dc.apply(lambda x: ",".join([_.replace("../data/pngs/", "") for _ in x.split(",")]))
prev_df = pd.read_csv("../../data/train_fracture_using_bbox_2dc_kfold.csv")
fold_cols = [c for c in prev_df.columns if "inner" in c or "outer" in c]
df[fold_cols] = -1
df = pd.concat([df, prev_df])
df.to_csv("../../data/train_fracture_pseudoslices_fold0_kfold.csv", index=False)