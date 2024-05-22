import cv2
import os.path as osp
import numpy as np
import pandas as pd
import pickle

from collections import defaultdict
from tqdm import tqdm

from utils import create_dir


def add_buffer(x1, x2, max_dist, buff=0.1):
    add_dist = int(buff * (x2 - x1))
    x1, x2 = x1 - add_dist, x2 + add_dist
    x1 = max(0, x1)
    x2 = min(max_dist, x2)
    return x1, x2


SAVE_DIR = "../../data/train-numpy-vertebra-chunks-with-20-buffer/"
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
    Z, H, W = img_array.shape
    for level in [f"C{i+1}" for i in range(7)]:
        filename = f"{study_id}_{level}.npy"
        z1, z2, h1, h2, w1, w2 = study_coords[level]
        z1, z2 = add_buffer(z1, z2, max_dist=Z, buff=0.25)
        h1, h2 = add_buffer(h1, h2, max_dist=H, buff=0.25)
        w1, w2 = add_buffer(w1, w2, max_dist=W, buff=0.25)
        img_chunk = img_array[z1:z2, h1:h2, w1:w2]
        np.save(osp.join(SAVE_DIR, filename), img_chunk)
        chunk_dict["filename"].append(filename)