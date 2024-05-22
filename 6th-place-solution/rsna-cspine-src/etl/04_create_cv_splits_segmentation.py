import glob
import pandas as pd

from utils import create_double_cv

studies = glob.glob("../../data/train-numpy-seg-blocks-128/*")
labels = [s.replace("train", "label") for s in studies]

df = pd.DataFrame({"filename": studies, "label": labels})
df["filename"] = df.filename.apply(lambda x: x.replace("../../data/", ""))
df["label"] = df.label.apply(lambda x: x.replace("../../data/", ""))

df["StudyInstanceUID"] = df.filename.apply(lambda x: x.split("/")[-1].split("-")[0])
df = create_double_cv(df, "StudyInstanceUID", 5, 5)

df.to_csv("../../data/train_seg_blocks_kfold.csv", index=False)