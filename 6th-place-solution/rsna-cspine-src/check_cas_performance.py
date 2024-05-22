import glob
import numpy as np
import pandas as pd

from functools import partial
from sklearn.metrics import fbeta_score, roc_auc_score


def find_optimal_threshold(y_true, y_pred, metric_func, thresholds=np.arange(0.5, 0.95, 0.05)):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metric_list = []
    for thresh in thresholds:
        metric_list.append(metric_func(y_true, y_pred >= thresh))
    print(f"BEST : {np.max(metric_list):0.3f} @ {thresholds[np.argmax(metric_list)]:0.2f}")


f2_score = partial(fbeta_score, beta=2)

bbox_df = pd.read_csv("../data/train_bounding_boxes.csv")
folds_df = pd.read_csv("../data/train_kfold.csv")
train_df = pd.read_csv("../data/train.csv")

positives_without_boxes = list(set(train_df[train_df.patient_overall == 1].StudyInstanceUID) - set(bbox_df.StudyInstanceUID))

bbox_df["study_slice"] = bbox_df.StudyInstanceUID + "_" + bbox_df.slice_number.astype("str")
cas_dfs = np.sort(glob.glob("../data/train_cas_fold*.csv"))
cas_dfs = [pd.read_csv(_) for _ in cas_dfs]
for each_cas_df in cas_dfs:
    each_cas_df["slice_number"] = each_cas_df.filename.apply(lambda x: x.split("/")[-1].split(".")[0])
    each_cas_df["study_slice"] = each_cas_df.StudyInstanceUID + "_" + each_cas_df.slice_number

for i, each_cas_df in enumerate(cas_dfs):
    each_cas_df = each_cas_df.merge(folds_df, on="StudyInstanceUID")
    oof_df = each_cas_df[each_cas_df.outer == i]
    # Exclude positives not in bbox_df
    oof_df = oof_df[~oof_df.StudyInstanceUID.isin(positives_without_boxes)]
    oof_df["y_true"] = 0
    oof_df.loc[oof_df.study_slice.isin(bbox_df.study_slice.tolist()), "y_true"] = 1
    print(f"FOLD{i}\n=====")
    print(oof_df.y_true.value_counts())
    print(f"AUC (all):     {roc_auc_score(oof_df.y_true, oof_df.cas):0.3f}")
    # Only include C-spine
    oof_df["cspine_present"] = oof_df[[f"C{i+1}_present" for i in range(7)]].sum(1)
    oof_df = oof_df[oof_df.cspine_present > 0]
    print(f"AUC (C-spine): {roc_auc_score(oof_df.y_true, oof_df.cas):0.3f}")
    print(f"--\nMEAN CAS POSITIVES : {oof_df[oof_df.y_true == 1].cas.mean():0.3f}")
    print(f"MEAN CAS NEGATIVES : {oof_df[oof_df.y_true == 0].cas.mean():0.3f}")
    find_optimal_threshold(oof_df.y_true, oof_df.cas, f2_score)

