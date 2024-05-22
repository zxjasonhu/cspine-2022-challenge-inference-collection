import os.path as osp
import pandas as pd
import pickle

from utils import create_dir


SAVE_DIR = "../../data/train-casseq-dual-labels/"
create_dir(SAVE_DIR)
df = pd.read_csv("../../data/train_cas_kfold_all_by_level.csv")
seq_df = pd.read_csv("../../data/train_seq_kfold.csv")
seq_dict = {study_id: study_df[[f"C{i+1}" for i in range(7)] + ["patient_overall"]].values
            for study_id, study_df in seq_df.groupby("StudyInstanceUID")}

for study_id, study_df in df.groupby("StudyInstanceUID"):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)

    with open(osp.join(SAVE_DIR, f"{study_id}.pkl"), "wb") as f:
        pickle.dump({"sequence": study_df.fracture_cas.values, "exam": seq_dict[study_id][0]}, f)

seq_df["label"] = seq_df.StudyInstanceUID.apply(lambda x: osp.join(SAVE_DIR.replace("../../data/", "../data/"),
                                                                   x + ".pkl"))
seq_df.to_csv("../../data/train_dual_seq_kfold.csv", index=False)