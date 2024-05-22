import cv2
import os.path as osp
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.insert(0, "../../src/")
import torch

from collections import defaultdict
from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from skp import builder
from tqdm import tqdm
from utils import create_dir


torch.set_grad_enabled(False)

CONFIG = "../configs/cascrop/cascrop007.yaml"
base_name = osp.basename(CONFIG).replace(".yaml", "")
SAVE_DIR = f"../../data/train-{base_name}-chunk-features/"

create_dir(SAVE_DIR)

cfg = OmegaConf.load(CONFIG)
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")

checkpoints = {
    f"fold{i}": f"../../experiments/{base_name}/sbn/fold{i}/checkpoints/best.ckpt" for i in range(5)
}

models = {}
for fold, ckpt in checkpoints.items():
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models[fold] = tmp_model
    create_dir(osp.join(SAVE_DIR, fold))

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
    img_array = np.stack([cv2.imread(osp.join("../../data/pngs-with-seg/", f)) for f in files])
    Z, H, W = img_array.shape[:3]
    for level in [f"C{i+1}" for i in range(7)]:
        filename = f"{study_id}_{level}.npy"
        z1, z2, h1, h2, w1, w2 = study_coords[level]
        img_chunk = img_array[z1:z2+1, h1:h2+1, w1:w2+1]
        img_chunk = zoom(img_chunk, [64 / img_chunk.shape[0], 288 / img_chunk.shape[1], 288 / img_chunk.shape[2], 1],
                         order=1, prefilter=False)
        assert img_chunk.shape == (64, 288, 288, 3)
        img_chunk = torch.from_numpy(img_chunk).float().cuda()
        img_chunk = img_chunk / 255.0
        img_chunk = img_chunk - 0.5
        img_chunk = img_chunk * 2.0
        img_chunk = img_chunk.permute(0, 3, 1, 2)
        for fold, model in models.items():
            features = model.extract_features(img_chunk)
            np.save(osp.join(SAVE_DIR, fold, filename), features.cpu().numpy())