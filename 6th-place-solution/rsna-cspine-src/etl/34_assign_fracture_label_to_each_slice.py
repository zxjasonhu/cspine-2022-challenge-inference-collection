import numpy as np
import pandas as pd
import pickle

from collections import defaultdict
from utils import convert_to_2dc


train_df = pd.read_csv("../../data/train.csv")
positive_studies = train_df[train_df.patient_overall == 1].StudyInstanceUID.tolist()
negative_studies = list(set(train_df.StudyInstanceUID) - set(positive_studies))
print(f"POSITIVES : N={len(positive_studies)}")
print(f"NEGATIVES : N={len(negative_studies)}")

folds_df = pd.read_csv("../../data/train_kfold.csv")
bbox_df = pd.read_csv("../../data/train_bounding_boxes.csv")
bbox_df["study_slice"] = bbox_df.StudyInstanceUID + "_" + bbox_df.slice_number.astype("str")
positive_studies_with_bbox = list(set(positive_studies) & set(bbox_df.StudyInstanceUID))

with open("../../data/train_vertebra_level_dict.pkl", "rb") as f:
    levels = pickle.load(f)

level_map_dict = {36: 1, 72: 2, 109: 3, 145: 4, 182: 5, 218: 6, 255: 7}
all_filenames = []
all_levels = []
for k, v in levels.items():
    all_filenames.extend(v["filenames"])
    tmp_levels = []
    for i in v["level"]:
        store_levels = [0, 0, 0, 0, 0, 0, 0]
        for j in i:
            if j == 0:
                continue
            store_levels[level_map_dict[j]-1] = 1
        all_levels.append(store_levels)

levels_arr = np.vstack(all_levels)
df = pd.DataFrame(levels_arr)
df.columns = [f"C{i+1}_present" for i in range(7)]
df["filename"] = all_filenames
df["StudyInstanceUID"] = df.filename.apply(lambda x: x.split("/")[0])
df["slice_number"] = df.filename.apply(lambda x: int(x.split("/")[-1].replace(".png", "")))
df["study_slice"] = df.StudyInstanceUID + "_" + df.slice_number.astype("str")
df = df[df.StudyInstanceUID.isin(positive_studies_with_bbox + negative_studies)]


df["fracture"] = 0
df.loc[df.study_slice.isin(bbox_df.study_slice.tolist()), "fracture"] = 1
print(df.fracture.value_counts())

# Assign level to each fracture
fractured_levels_for_each_study = defaultdict(list)
for _, row in train_df[train_df.StudyInstanceUID.isin(bbox_df.StudyInstanceUID.tolist())].iterrows():
    for level in [f"C{i+1}" for i in range(7)]:
        if row[level] == 1:
            fractured_levels_for_each_study[row.StudyInstanceUID].append(level)

for level in [f"C{i+1}_frac" for i in range(7)]:
    df[level] = 0

pos_df = df[df.fracture == 1].reset_index(drop=True)
neg_df = df[df.fracture == 0].reset_index(drop=True)

for row_index, row in pos_df.iterrows():
    fractured_levels = fractured_levels_for_each_study[row.StudyInstanceUID]
    for level in fractured_levels:
        if row[f"{level}_present"]:
            pos_df.loc[row_index, f"{level}_frac"] = 1

df = pd.concat([pos_df, neg_df])
del pos_df
del neg_df

for level in [f"C{i+1}_frac" for i in range(7)]:
    print(df[level].value_counts())

study_level_df = df[["StudyInstanceUID"]].merge(train_df[["StudyInstanceUID", "patient_overall"]]).drop_duplicates()
print(study_level_df.patient_overall.value_counts())

df = df.merge(folds_df, on="StudyInstanceUID")
df = df.sort_values(["StudyInstanceUID", "slice_number"])
df.to_csv("../../data/train_cspine_crunchtime.csv", index=False)

