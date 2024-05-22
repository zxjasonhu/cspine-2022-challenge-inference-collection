import numpy as np
import pandas as pd

from utils import create_double_cv


df = pd.read_csv("../../data/train_metadata_2dc.csv")
cas_df = pd.read_csv("../../data/train_cas.csv")
cas_dict = {row.filename: row.cas for row in cas_df.itertuples()}
df_cas_list = []
for row in df.itertuples():
    if row.filename in cas_dict:
        df_cas_list.append(cas_dict[row.filename.replace("png", "dcm")])
    else:
        df_cas_list.append(0)

df["cas"] = df_cas_list
df["cas"] /= df.cas.max()

df["fracture_cas"] = (df.cas >= 0.2).astype("int")
print(df.fracture_cas.value_counts())
fractures_by_study = [study_df.fracture_cas.max() for _, study_df in df.groupby("StudyInstanceUID")]
print(np.unique(fractures_by_study, return_counts=True))

df = create_double_cv(df, "StudyInstanceUID", 5, 5)
df["filename"] = df.filename.apply(lambda x: x.replace("dcm", "png"))
df.to_csv("../../data/train_cas_2dc_kfold.csv", index=False)


import pandas as pd

from utils import convert_to_2dc


df = pd.read_csv("../../data/train_cas_kfold_all_by_level.csv")
new_df_list = []
for study_id, study_df in df.groupby("StudyInstanceUID"):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    study_df["filename_2dc"] = convert_to_2dc(study_df.filename.tolist(), 5)
    new_df_list.append(study_df)


new_df = pd.concat(new_df_list)
new_df.to_csv("../../data/train_cas_2dc-5_kfold_all_by_level.csv", index=False)