import cv2
import glob
import os.path as osp
import numpy as np
import pandas as pd
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

CONFIG = "../configs/caschunk/caschunk001.yaml"
base_name = osp.basename(CONFIG).replace(".yaml", "")
SAVE_DIR = f"../../data/train-{base_name}-features/"
CHUNK_FEATURES_DIR = f"../../data/train-cascrop007-chunk-features/"

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

study_ids = glob.glob(osp.join(CHUNK_FEATURES_DIR, "fold*/*.npy"))
study_ids = np.unique([_.split("/")[-1].split("_")[0] for _ in study_ids])

for each_study in tqdm(study_ids, total=len(study_ids)):
    for fold in [*models]:
        features = [np.load(_) for _ in np.sort(glob.glob(osp.join(CHUNK_FEATURES_DIR, fold, f"{each_study}_C*")))]
        features_list = []
        for feat in features:
            feat = torch.from_numpy(feat).unsqueeze(0).cuda()
            mask = torch.ones(feat.shape[:2]).cuda()
            level_feature = models[fold].extract_features((feat, mask))
            features_list.append(level_feature.cpu().numpy())
        new_features = np.concatenate(features_list)
        assert new_features.shape[0] == 7
        np.save(osp.join(SAVE_DIR, fold, f"{each_study}.npy"), new_features)