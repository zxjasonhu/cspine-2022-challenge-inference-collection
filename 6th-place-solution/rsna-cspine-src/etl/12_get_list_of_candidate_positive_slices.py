"""
Based on cervical spine segmentations, extract slices which contain vertebra levels
that are positive for fracture, based on the study-level labels.

Not all of these slices will necessarily be positive for fracture, so this is where
pseudolabeling will help to further refine these labels.
"""
import numpy as np
import pandas as pd

df = pd.read_csv("../../data/train_metadata_with_cspine_level.csv")
bbox_df = pd.read_csv("../../data/train_bounding_boxes.csv")
train_df = pd.read_csv("../../data/train.csv")

# First, just exclude everything without C-spine
df["cspine_present"] = df[[f"C{i+1}_present" for i in range(7)]].sum(axis=1)
df = df[df.cspine_present > 0]

# Then, exclude studies with bounding boxes, since we already know the slice-level fracture
# levels for those
df = df[~df.StudyInstanceUID.isin(bbox_df.StudyInstanceUID.tolist())]

# Then, exclude negative studies
negative_studies = train_df[train_df.patient_overall == 0].StudyInstanceUID.tolist()
df = df[~df.StudyInstanceUID.isin(negative_studies)]

# For each study, find out which vertebra levels are fractured and only retain the slices
# from those levels
df_list = []
level_fracture_cols = [f"C{i+1}" for i in range(7)]
for study_id, study_df in df.groupby("StudyInstanceUID"):
    study_labels = train_df[train_df.StudyInstanceUID == study_id]
    pos_levels = (study_labels[level_fracture_cols] == 1).values[0]
    pos_levels = np.asarray(level_fracture_cols)[pos_levels]
    levels_with_fx = []
    for level in pos_levels:
        levels_with_fx.append(study_df[study_df[f"{level}_present"] == 1])
    levels_with_fx = pd.concat(levels_with_fx)
    df_list.append(levels_with_fx)

levels_with_fx_df = pd.concat(df_list)

with open("../positive_fracture_candidate_slices.txt", "w") as f:
    for filename in levels_with_fx_df.filename:
        _ = f.write(f"{filename.replace('dcm', 'png')}\n")

