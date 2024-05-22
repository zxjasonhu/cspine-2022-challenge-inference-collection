import pandas as pd
import pickle

from tqdm import tqdm


df = pd.read_csv("../../data/train_metadata.csv")
with open("../../data/train_cspine_coords.pkl", "rb") as f:
    cspine_coords = pickle.load(f)

df_list = []
for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False).reset_index(drop=True)
    study_cspine_coords = cspine_coords[study_id]
    for level in [f"C{i+1}" for i in range(7)]:
        level_coords = study_cspine_coords[level]
        z1, z2 = level_coords[:2]
        study_df[f"{level}_present"] = 0
        study_df.loc[z1:z2, f"{level}_present"] = 1
    df_list.append(study_df)

df = pd.concat(df_list)
df.to_csv("../../data/train_metadata_with_cspine_level.csv", index=False)

