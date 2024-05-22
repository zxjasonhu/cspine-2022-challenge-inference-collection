import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd
import torch

from etl.utils import create_dir
from omegaconf import OmegaConf
from skp import builder 
from tqdm import tqdm


torch.set_grad_enabled(False)

SAVE_FEATURES_DIR = "../data/train-cas001-features/"

checkpoints = np.sort(glob.glob("../experiments/cas001/sbn/fold*/checkpoints/best.ckpt"))
config = OmegaConf.load("configs/cas/cas001.yaml")

models = {}
for ckpt in checkpoints:
  config.model.load_pretrained = str(ckpt)
  mod = builder.build_model(config)
  mod = mod.cuda().eval()
  models[ckpt.split("/")[-3]] = mod

print(models.keys())
for fold in [*models]:
  create_dir(osp.join(SAVE_FEATURES_DIR, fold))

df = pd.read_csv("../data/train_cas_kfold_all_by_level.csv")

dataset = builder.build_dataset(config, {"inputs": [0], "labels": [0]}, mode="predict")

for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
  study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
  images = study_df.filename.tolist()
  images = [im.replace("dcm", "png") for im in images]
  images = [osp.join("../data/pngs-with-seg/", im) for im in images]
  X = np.stack([dataset.process_image({"image": cv2.imread(im)})["image"] for im in images])
  X = torch.from_numpy(X).float().cuda()
  for fold, mod in models.items():
    features = np.concatenate([mod.extract_features(X[batch:batch+16]).cpu().numpy() for batch in range(0, len(X), 16)])
    np.save(osp.join(SAVE_FEATURES_DIR, fold, study_id + ".npy"), features)
