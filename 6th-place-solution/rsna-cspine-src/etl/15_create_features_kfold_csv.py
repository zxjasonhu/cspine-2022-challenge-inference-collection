import glob
import pandas as pd


df = pd.read_csv("../../data/train_vertebra_chunks_kfold.csv")
train_df = pd.read_csv("../../data/train.csv")

features = glob.glob("../../data/chunk-features/fold0/*.npy")
feat_df = pd.DataFrame({"filename": features})
feat_df["filename"] = feat_df.filename.apply(lambda x: x.split("/")[-1])
feat_df["StudyInstanceUID"] = feat_df.filename.apply(lambda x: x.replace(".npy", ""))

feat_df = feat_df.merge(df[["StudyInstanceUID"] + [c for c in df.columns if "outer" in c or "inner" in c]])
feat_df = feat_df.drop_duplicates()
feat_df = feat_df.merge(train_df, on="StudyInstanceUID")
feat_df.to_csv("../../data/train_chunk_features_kfold.csv", index=False)