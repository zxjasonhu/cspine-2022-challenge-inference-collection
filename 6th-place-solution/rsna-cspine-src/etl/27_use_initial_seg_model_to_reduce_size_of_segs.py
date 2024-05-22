import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd
import pickle
import torch

from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

import sys
sys.path.insert(0, "../../src/")

from skp import builder
from utils import create_dir


def predict(model, X):
    X = torch.from_numpy(X).float().cuda().unsqueeze(0).unsqueeze(0)
    return torch.sigmoid(model(X)).cpu()


torch.set_grad_enabled(False)

IMSIZE = 192.
CONFIG = "../configs/seg/pseudoseg000.yaml"
SAVE_CROPPED_DIR = "../../data/train-cropped-cspine-volumes/"
SAVE_CROPPED_ORIG_SEG = "../../data/train-cropped-orig-seg/"
create_dir(SAVE_CROPPED_DIR)
create_dir(SAVE_CROPPED_ORIG_SEG)

existing_segmentations = glob.glob("../../data/segmentations-numpy/*.npy")
existing_segmentations = [_.split("/")[-1].replace(".npy", "") for _ in existing_segmentations]

df = pd.read_csv("../../data/train_metadata.csv")
cfg = OmegaConf.load(CONFIG)
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")
checkpoints = np.sort(glob.glob(f"../../experiments/{osp.basename(CONFIG).replace('.yaml', '')}/sbn/fold*/checkpoints"
                                f"/best.ckpt"))
models = []
for ckpt in checkpoints:
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models.append(tmp_model)

THRESHOLD = 0.1
coords_dict = {}
for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    study_images = study_df.filename.tolist()
    study_images = [osp.join("../../data/pngs", i.replace("dcm", "png")) for i in study_images]
    orig_X = np.stack([cv2.imread(im, 0) for im in study_images])
    z, h, w = orig_X.shape
    rescale_factor = [IMSIZE / z, IMSIZE / h, IMSIZE / w]
    X = zoom(orig_X, rescale_factor, order=0, prefilter=False)
    X = dataset.preprocess(X)
    p = torch.cat([predict(m, X) for m in models]).mean(0)
    assert p.size(0) == 8
    p_spine = p[:7].sum(0)
    spine_coords = np.vstack(np.where(p_spine.numpy() >= THRESHOLD)).astype("float")
    for axis in range(3):
        spine_coords[axis] /= rescale_factor[axis]
    spine_coords = spine_coords.astype("int")
    try:
        z1, z2, h1, h2, w1, w2 = spine_coords[0].min(), spine_coords[0].max(), spine_coords[1].min(), spine_coords[1].max(),\
                                 spine_coords[2].min(), spine_coords[2].max()
    except Exception as e:
        print(f"{study_id} FAILED !!")
        continue

    cropped_X = orig_X[z1:z2+1, h1:h2+1, w1:w2+1]
    print(f"{orig_X.shape} --> {cropped_X.shape}")
    np.save(osp.join(SAVE_CROPPED_DIR, f"{study_id}.npy"), cropped_X)
    if study_id in existing_segmentations:
        orig_seg = np.load(osp.join("../../data/segmentations-numpy/", f"{study_id}.npy"))
        np.save(osp.join(SAVE_CROPPED_ORIG_SEG, f"{study_id}.npy"), orig_seg[z1:z2+1, h1:h2+1, w1:w2+1])