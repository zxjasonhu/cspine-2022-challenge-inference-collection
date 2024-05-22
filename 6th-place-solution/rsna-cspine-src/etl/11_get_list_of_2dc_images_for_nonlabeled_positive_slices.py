import pandas as pd

from utils import create_double_cv


df = pd.read_csv("../../data/train.csv")
level_df = pd.read_csv("../../data/train_slices_with_vertebra_levels.csv")
level_df["cspine_present"] = level_df[[f"C{i+1}" for i in range(7)]].sum(axis=1)
level_df = level_df[level_df.cspine_present > 0]
bbox_df = pd.read_csv("../../data/train_bounding_boxes.csv")

df = df.merge(level_df, on="StudyInstanceUID")
df["slice_number"] = df.filename.apply(lambda x: x.split("/")[-1].replace(".dcm", "")).astype("int")
df["fracture"] = 0

pos_df = df[df.patient_overall == 1]
pos_df = pos_df[~pos_df.StudyInstanceUID.isin(bbox_df.StudyInstanceUID.tolist())]

df_list = []
for study_id, study_df in pos_df.groupby("StudyInstanceUID"):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    mid = [f.replace("dcm", "png") for f in study_df.filename]
    top = [mid[0]] + mid[:-1]
    bot = mid[1:] + [mid[-1]]
    filename_2dc = [f"{t},{m},{b}" for t,m,b in zip(top,mid,bot)]
    study_df["filename_2dc"] = filename_2dc
    df_list.append(study_df)

df = pd.concat(df_list)

with open("../unlabeled_positive_slices_2dc.txt", "w") as f:
    for fi in df.filename_2dc:
        _ = f.write(f"{fi}\n")

