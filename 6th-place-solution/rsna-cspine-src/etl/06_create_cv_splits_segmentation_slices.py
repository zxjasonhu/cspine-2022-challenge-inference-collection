import glob
import numpy as np
import pandas as pd

from utils import create_double_cv

studies = glob.glob("../../data/train-numpy-seg-pngs/*")
labels = [s.replace("train", "label") for s in studies]

df = pd.DataFrame({"filename": studies, "label": labels})
df["filename"] = df.filename.apply(lambda x: x.replace("../../data/", ""))
df["label"] = df.label.apply(lambda x: x.replace("../../data/", ""))
df["slice_number"] = df.filename.apply(lambda x: x.split("/")[-1].split("-")[-1].replace(".png", "")).astype("int")
df["StudyInstanceUID"] = df.filename.apply(lambda x: x.split("/")[-1].split("-")[0])
df = pd.concat([_df.sort_values("slice_number", ascending=True) for study_id, _df in df.groupby("StudyInstanceUID")]).reset_index(drop=True)
df_list = []
for study_id, _df in df.groupby("StudyInstanceUID"):
    _df = _df.sort_values("slice_number", ascending=True)
    filenames = _df.filename.tolist()
    top = [filenames[0]] + filenames[:-1]
    bot = filenames[1:] + [filenames[-1]]
    files_2dc = [f"{t},{m},{b}" for t,m,b in zip(top, filenames, bot)]
    _df["filename_2dc"] = files_2dc
    df_list.append(_df)


df = pd.concat(df_list).reset_index(drop=True)
print(df.filename_2dc.values[3], df.label.values[3])
df = create_double_cv(df, "StudyInstanceUID", 5, 5)

df.to_csv("../../data/.csv", index=False)