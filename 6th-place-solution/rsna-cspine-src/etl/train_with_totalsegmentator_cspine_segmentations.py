import glob
import pandas as pd


folds_df = pd.read_csv("../../data/train_kfold.csv")
totalsegmentator_segmentations = glob.glob("../../data/totalsegmentator-cspine-segmentations/*.npy")
segmentations = glob.glob("../../data/label-numpy-seg-whole-192x192x192/*")
segmentations = [s.split("/")[-1].replace(".npy", "") for s in segmentations]

df = pd.DataFrame({"label": totalsegmentator_segmentations})
df["label"] = df.label.apply(lambda x: x.replace("../../data/", ""))
df["StudyInstanceUID"] = df.label.apply(lambda x: x.split("/")[-1].replace(".npy", ""))
df["filename"] = "train-pseudo-seg-numpy/" + df.StudyInstanceUID + ".npy"
df = df.merge(folds_df, on="StudyInstanceUID")
#df.loc[~df.StudyInstanceUID.isin(segmentations), "outer"] = -1
print(df.outer.value_counts())

df.to_csv("../../data/train_totalsegmentator_cspine.csv", index=False)


