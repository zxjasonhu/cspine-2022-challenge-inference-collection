import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd
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

IMAGE_SAVE_DIR = "../../data/pngs-with-seg/"
create_dir(IMAGE_SAVE_DIR)

IMSIZE = 192.

df = pd.read_csv("../../data/train_metadata.csv")

cfg = OmegaConf.load("../configs/seg/pseudoseg000.yaml")
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")
checkpoints = np.sort(glob.glob("../../experiments/pseudoseg000/sbn/fold*/checkpoints/last.ckpt"))
models = []
for ckpt in checkpoints:
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models.append(tmp_model)

for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    study_images = study_df.filename.tolist()
    study_images = [osp.join("../../data/pngs", i.replace("dcm", "png")) for i in study_images]
    X_orig = np.stack([cv2.imread(im, 0) for im in study_images])
    z, h, w = X_orig.shape
    rescale_factor = [IMSIZE / z, IMSIZE / h, IMSIZE / w]
    X = zoom(X_orig, rescale_factor, order=0, prefilter=False)
    X = dataset.preprocess(X)
    p = torch.cat([predict(m, X) for m in models]).mean(0)
    p_spine = p.sum(0)
    p = torch.argmax(p, dim=0) + 1
    p[p_spine < 0.4] = 0
    p[p == 8] = 0
    p = torch.nn.functional.interpolate(p.unsqueeze(0).unsqueeze(0).float(), size=(z, h, w), mode="nearest")
    p = p.squeeze(0).squeeze(0)
    p = p.numpy()
    assert np.max(p) <= 7
    p = (p * 255) / 7
    p = p.astype("uint8")
    X_orig = np.stack([X_orig, X_orig, p], axis=-1)
    tmp_save_dir = osp.join(IMAGE_SAVE_DIR, study_id)
    create_dir(tmp_save_dir)
    for ind, each_img in enumerate(X_orig):
        _ = cv2.imwrite(study_images[ind].replace("../../data/pngs", IMAGE_SAVE_DIR), each_img)

