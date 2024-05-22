import glob
import numpy as np
import pandas as pd
import pickle
import torch

from sklearn.metrics import roc_auc_score


def sensitivity(t, p):
    t, p = np.asarray(t), np.asarray(p)
    tp = np.sum(t + p == 2)
    return tp / np.sum(t)


def specificity(t, p):
    t, p = np.asarray(t), np.asarray(p)
    tn = np.sum(t + p == 0)
    return tn / np.sum(t == 0)


pickle_files = np.sort(glob.glob("../predictions/cls003/fold*pkl"))
test_df = pd.read_csv("../data/train_2dc_classes_kfold.csv")

pred_df_list = []
for each_pickle in pickle_files:
    with open(each_pickle, "rb") as f:
        preds = pickle.load(f)
    p = torch.sigmoid(torch.cat([preds[i][0] for i in range(len(preds))], dim=0)).cpu().numpy()
    names = np.concatenate([preds[i][1] for i in range(len(preds))], axis=0)
    pred_df = pd.DataFrame(p)
    pred_df.columns = ["lb_pred", "sb_pred", "st_pred", "gi_pred"]
    pred_df["name"] = names
    pred_df_list.append(pred_df)

pred_df = pd.concat(pred_df_list)
pred_df["filename"] = pred_df["name"].apply(lambda x: x.split(",")[1].replace("../data/", ""))
pred_df = pred_df.merge(test_df, on="filename")
roc_auc_score(pred_df.gi_tract.values, pred_df.gi_pred.values)

df = pd.read_csv("../data/train_2dc_kfold.csv")

df = df.merge(pred_df[["gi_pred", "filename"]], on="filename")

THRESHOLD = 0.025
y_true, y_pred = [], []
df_list = []
for case_day, _df in pred_df.groupby("case_day"):
    _df = _df.sort_values("slice_id", ascending=True)
    positive = _df.gi_pred.values >= THRESHOLD 
    positive = np.where(positive)[0]
    start, stop = positive[0], positive[-1]
    start, stop = max(0, start-5), min(len(_df), stop+5)
    y_true.extend(_df.gi_tract.tolist())
    p = np.zeros((len(_df),))
    p[start:stop+1] = 1
    y_pred.extend(list(p))
    _df.gi_pred = p
    df_list.append(_df)

df = pd.concat(df_list)

sensitivity(df.gi_tract.values, df.gi_pred.values)
specificity(df.gi_tract.values, df.gi_pred.values)

df.to_csv("../data/train_2dc_kfold_with_slice_preds.csv", index=False)

sensitivity(y_true, y_pred)
specificity(y_true, y_pred)

sensitivity(pred_df.gi_tract.values, (pred_df.iloc[:,3] >= 0.3).astype("int"))