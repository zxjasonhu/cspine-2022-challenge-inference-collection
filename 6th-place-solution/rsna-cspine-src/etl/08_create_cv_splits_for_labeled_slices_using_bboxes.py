import pandas as pd

from utils import create_double_cv


level_df = pd.read_csv("../../data/train_slices_with_vertebra_levels.csv")
bbox_df = pd.read_csv("../../data/train_bounding_boxes.csv")
train_df = pd.read_csv("../../data/train.csv")
negative_studies = train_df.loc[train_df.patient_overall == 0, "StudyInstanceUID"]

#level_df = level_df[level_df.StudyInstanceUID.isin(bbox_df.StudyInstanceUID.tolist())]
#level_df["c_spine_present"] = level_df[[f"C{i+1}" for i in range(7)]].sum(axis=1) > 0
#level_df = level_df[level_df.c_spine_present]
level_df["slice_number"] = level_df.filename.apply(lambda x: int(x.split("/")[-1].replace(".dcm", "")))
level_df["study_slice"] = level_df.StudyInstanceUID + "_" + level_df.slice_number.astype("str")

bbox_df["study_slice"] = bbox_df.StudyInstanceUID + "_" + bbox_df.slice_number.astype("str")

level_df["fracture"] = 0
level_df.loc[level_df.study_slice.isin(bbox_df.study_slice.tolist()), "fracture"] = 1
# Positive slices = annotated slices with bounding boxes
#   NOTE: negative slices from these studies are EXCLUDED
# Negative slices come from negative STUDIES
pos_df = level_df[level_df.fracture == 1]
neg_df = level_df[level_df.StudyInstanceUID.isin(negative_studies.tolist())]
neg_df = neg_df.sort_values(["StudyInstanceUID", "ImagePositionPatient_2"]).reset_index(drop=True)
neg_df = neg_df.iloc[::4]
level_df = pd.concat([pos_df, neg_df])

print(f"Percent positive fracture slices : {level_df.fracture.mean() * 100:0.1f}%")
print(level_df.shape)
print(level_df.fracture.value_counts())
level_df["filename"] = level_df.filename.apply(lambda x: x.replace("dcm", "png"))

level_df = create_double_cv(level_df, "StudyInstanceUID", 5, 5)
level_df.to_csv("../../data/train_fracture_using_bbox_2dc_kfold.csv", index=False)