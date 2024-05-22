import cv2
import os.path as osp
import numpy as np
import pandas as pd
import pickle

from collections import defaultdict
from tqdm import tqdm

from utils import create_dir


SAVE_DIR = "../../data/train-numpy-full-cspine/"
create_dir(SAVE_DIR)

df = pd.read_csv("../../data/train_metadata.csv")
train_df = pd.read_csv("../../data/train.csv")
folds_df = pd.read_csv("../../data/train_kfold.csv")
with open("../../data/train_cspine_coords.pkl", "rb") as f:
    coords_dict = pickle.load(f)

chunk_dict = defaultdict(list)
for study_id, study_coords in tqdm(coords_dict.items(), total=len(coords_dict)):
    study_df = df[df.StudyInstanceUID == study_id]
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    files = study_df.filename.apply(lambda x: x.replace("dcm", "png")).tolist()
    img_array = np.stack([cv2.imread(osp.join("../../data/pngs", f), 0) for f in files])
    coords_list = []
    for level in [f"C{i+1}" for i in range(7)]:
        coords_list.append(study_coords[level])
    coords_arr = np.vstack(coords_list).swapaxes(0, 1)
    z1, z2, h1, h2, w1, w2 = coords_arr[0].min(), coords_arr[1].max(), coords_arr[2].min(), coords_arr[3].max(), coords_arr[4].min(), coords_arr[5].max()
    img_array = img_array[z1:z2, h1:h2, w1:w2]
    np.save(osp.join(SAVE_DIR, f"{study_id}.npy"), img_array)


