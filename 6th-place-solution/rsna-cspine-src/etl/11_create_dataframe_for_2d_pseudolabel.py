import numpy as np
import pandas as pd


df = pd.read_csv("../../data/train_metadata_with_cspine_level.csv")
train_df = pd.read_csv("../../data/train.csv")
df = df.merge(train_df[["StudyInstanceUID", "patient_overall"]], on="StudyInstanceUID")
bbox_df = pd.read_csv("../../data/train_bounding_boxes.csv")

# Exclude positive studies WITHOUT bounding boxes
pos_studies_without_bbox = list(set(df[df.patient_overall == 1].StudyInstanceUID) - set(bbox_df.StudyInstanceUID))
df = df[~df.StudyInstanceUID.isin(pos_studies_without_bbox)]

df["slice_number"] = df.filename.apply(lambda x: x.split("/")[-1].split(".")[0])
df["study_slice"] = df.StudyInstanceUID + "_" + df.slice_number.astype("str")
bbox_df["study_slice"] = bbox_df.StudyInstanceUID + "_" + bbox_df.slice_number.astype("str")

df["cspine_present"] = df[[f"C{i+1}_present" for i in range(7)]].sum(axis=1)
df = df[df.cspine_present > 0]
df["fracture"] = 0
df.loc[df.study_slice.isin(bbox_df.study_slice.tolist()), "fracture"] = 1
print(df.fracture.value_counts())
print(df[["StudyInstanceUID", "patient_overall"]].drop_duplicates().patient_overall.value_counts())

# Randomly sample 20 positives and 30 negative studies to use as validation set during training
df["outer"] = -1
np.random.seed(88)
val_pos = np.random.choice(df[df.fracture == 1].StudyInstanceUID.unique(), 20, replace=False)
val_neg = np.random.choice(df[df.fracture == 0].StudyInstanceUID.unique(), 30, replace=False)
df.loc[df.StudyInstanceUID.isin(np.concatenate([val_pos, val_neg])), "outer"] = 0
print(df.outer.value_counts())
df["filename"] = df.filename.apply(lambda x: x.replace("dcm", "png"))

df.to_csv("../../data/train_initial_for_2d_pseudolabel_with_split.csv", index=False)
