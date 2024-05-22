import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score


df = pd.read_csv("../data/train_cas.csv")
bbox_df = pd.read_csv("../data/train_bounding_boxes.csv")

df["slice_number"] = df.filename.apply(lambda x: x.split("/")[-1].replace(".dcm", ""))
df["study_slice"] = df.StudyInstanceUID + "_" + df.slice_number.astype("str")
bbox_df["study_slice"] = bbox_df.StudyInstanceUID + "_" + bbox_df.slice_number.astype("str")

pos_df = df[df.StudyInstanceUID.isin(bbox_df.StudyInstanceUID.tolist())]
pos_df["fracture"] = 0
pos_df.loc[pos_df.study_slice.isin(bbox_df.study_slice.tolist()), "fracture"] = 1
pos_df.loc[pos_df.cas < 0, "cas"] = 0
pos_df["cas"] = pos_df.cas / pos_df.cas.max()

print(pos_df.fracture.value_counts())
print(roc_auc_score(pos_df.fracture, pos_df.cas))

for i in np.arange(0, 1, 0.05):
    print(f"THRESHOLD {i:.2f} : {f1_score(pos_df.fracture, (pos_df.cas >= i).astype('float')):0.3f}")
