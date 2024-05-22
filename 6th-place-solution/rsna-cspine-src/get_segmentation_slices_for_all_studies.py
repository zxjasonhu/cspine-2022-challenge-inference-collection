import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import torch

from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from etl.utils import create_dir
from skp import builder


torch.set_grad_enabled(False)

IMAGE_SAVEDIR = "../data/pngs-segmentations/"
IMSIZE = 192.

create_dir(IMAGE_SAVEDIR)

df = pd.read_csv("../data/train_metadata.csv")

cfg = OmegaConf.load("configs/mk3d/mk3d001.yaml")
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")
checkpoints = np.sort(glob.glob("../experiments/mk3d001/sbn/fold*/checkpoints/best.ckpt"))
models = []
for ckpt in checkpoints:
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg)
    tmp_model = tmp_model.eval().cuda()
    models.append(tmp_model)

for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    study_images = study_df.filename.tolist()
    study_images = [osp.join("../data/pngs", i.replace("dcm", "png")) for i in study_images]
    X = np.stack([cv2.imread(im, 0) for im in study_images])
    z, h, w = X.shape
    rescale_factor = [IMSIZE / z, IMSIZE / h, IMSIZE / w]
    X = zoom(X, rescale_factor, order=0, prefilter=False)
    X = dataset.preprocess(X)
    X = torch.from_numpy(X).float().cuda().unsqueeze(0).unsqueeze(0)
    p = torch.sigmoid(torch.stack([m(X).cpu() for m in models])).mean(0).squeeze(0)
    p = torch.cat([1 - p.sum(0).unsqueeze(0), p])
    p = torch.argmax(p, dim=0).float()
    p = torch.nn.functional.interpolate(p.unsqueeze(0).unsqueeze(0), size=(z, h, w), mode="nearest").squeeze(0).squeeze(0)
    p = p / 8.
    p = p * 255.
    p = p.numpy().astype("uint8")
    assert p.shape[0] == len(study_images)
    create_dir(osp.join(IMAGE_SAVEDIR, study_id))
    for ind, each_study_image in enumerate(study_images):
        filename = each_study_image.split("/")[-1].replace("dcm", "png")
        status = cv2.imwrite(osp.join(IMAGE_SAVEDIR, study_id, filename), p[ind])
