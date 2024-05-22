import cv2
import numpy as np
import pandas as pd
import os.path as osp
import pickle

from collections import defaultdict
from tqdm import tqdm
from utils import create_dir


SAVE_DIR = "../../data/pngs-cropped/"
create_dir(SAVE_DIR)

with open("../../data/train_cspine_coords.pkl", "rb") as f:
    cspine_coords = pickle.load(f)

df = pd.read_csv("../../data/train_metadata.csv")
cas_df = pd.read_csv("../../data/train_cas_kfold_all_by_level.csv")
used_files = []
for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    if study_id not in [*cspine_coords]:
        continue
    study_coords = cspine_coords[study_id]
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    files = study_df.filename.apply(lambda x: x.replace("dcm", "png")).values
    img_array = np.stack([cv2.imread(osp.join("../../data/pngs", f)) for f in files])
    Z, H, W = img_array.shape[:3]
    coords_list = []
    for level, level_coords in study_coords.items():
        coords_list.append(level_coords)
    coords_arr = np.vstack(coords_list)
    z1, z2, h1, h2, w1, w2 = coords_arr[0].min(), coords_arr[0].max(), \
                             coords_arr[1].min(), coords_arr[1].max(), \
                             coords_arr[2].min(), coords_arr[2].max()
    h1 = max(0, h1 - int(0.1 * (h2 - h1)))
    w1 = max(0, w1 - int(0.1 * (w2 - w1)))
    h2 = min(H, h2 + int(0.1 * (h2 - h1)))
    w2 = min(W, w2 + int(0.1 * (w2 - w1)))
    img_array = img_array[z1:z2, h1:h2, w1:w2]
    create_dir(osp.join(SAVE_DIR, study_id))
    for each_slice, each_file in zip(img_array, files[z1:z2]):
        status = cv2.imwrite(osp.join(SAVE_DIR, each_file), each_slice)
    break


