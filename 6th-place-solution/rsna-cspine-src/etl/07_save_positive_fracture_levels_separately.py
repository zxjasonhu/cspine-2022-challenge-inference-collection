import cv2
import glob
import os.path as osp
import pandas as pd

from tqdm import tqdm

from utils import create_dir


SAVEDIR = "../../data/positive-fractures-for-review/"
create_dir(SAVEDIR)

df = pd.read_csv("../../data/train.csv")
bbox_df = pd.read_csv("../../data/train_bounding_boxes.csv")
meta_df = pd.read_csv("../../data/train_metadata.csv")

df = df[~df.StudyInstanceUID.isin(bbox_df.StudyInstanceUID.tolist())]
pos_df = df[df.patient_overall == 1]
pos_df_columns = list(pos_df.columns)
pos_df_columns[2:] = [c + "_fracture" for c in pos_df_columns[2:]]
pos_df.columns = pos_df_columns
assert len(pos_df) == len(pos_df.StudyInstanceUID.unique())

levels_df = pd.read_csv("../../data/train_cspine_levels.csv")
pos_df = pos_df.merge(levels_df, on="StudyInstanceUID")

for study_id, study_df in tqdm(pos_df.groupby("StudyInstanceUID"), total=len(pos_df.StudyInstanceUID.unique())):
    create_dir(osp.join(SAVEDIR, study_id))
    for level in [f"C{i+1}" for i in range(6)]:
        if study_df[f"{level}_fracture"].values[0] == 1:
            files = study_df[study_df[level]].filename.tolist()
            images = [cv2.imread(osp.join("../../data/", f.replace("-segmentations", "")), 0) for f in files]
            slice_numbers = study_df[study_df[level]].slice_number.tolist()
            for im, slice_num in zip(images, slice_numbers):
                status = cv2.imwrite(osp.join(SAVEDIR, study_id, f"{level}_{slice_num:04d}.jpeg"), im)



