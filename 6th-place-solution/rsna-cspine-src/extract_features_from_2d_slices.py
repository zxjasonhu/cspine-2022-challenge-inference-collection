import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import torch

from collections import defaultdict
from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from skp import builder
from etl.utils import create_dir


def predict(model, X):
    X = torch.from_numpy(X).float().cuda().unsqueeze(0).unsqueeze(0)
    return torch.sigmoid(model(X)).cpu()


CONFIG = "configs/pre/pre200.yaml"
EXP_NAME = osp.basename(CONFIG).replace(".yaml", "")

SAVE_FEATURES_DIR = f"../data/train-{EXP_NAME}-features/"
create_dir(SAVE_FEATURES_DIR)

PNGS_DIR = "../data/train-individual-vertebrae-cropped-pngs/"

torch.set_grad_enabled(False)

PLOT = False


studies = glob.glob(osp.join(PNGS_DIR, "*"))

cfg = OmegaConf.load(CONFIG)
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")

checkpoints = {
    f"fold{i}": f"../experiments/{EXP_NAME}/sbn/fold{i}/checkpoints/best.ckpt" for i in range(3)
}

models = {}
for fold, ckpt in checkpoints.items():
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models[fold] = tmp_model
    create_dir(osp.join(SAVE_FEATURES_DIR, fold))

features_dict = defaultdict(list)
for study_folder in tqdm(studies):
    study_id = osp.basename(study_folder)
    study_images = glob.glob(osp.join(study_folder, "*.png"))
    images = np.asarray([cv2.imread(_) for _ in study_images])
    image_numbers = [int(_.split("/")[-1].replace(".png", "")) for _ in study_images]
    images = images[np.argsort(image_numbers)]
    images = np.stack([dataset.process_image({"image": _})["image"] for _ in images])
    images = torch.from_numpy(images).float().cuda()
    for fold in [*checkpoints]:
        features = torch.cat([models[fold].extract_features(images[_:_+32]) for _ in range(0, len(images), 32)]).cpu().numpy()
        np.save(osp.join(SAVE_FEATURES_DIR, fold, study_id + ".npy"), features)
    features_dict["filename"].append(study_id + ".npy")
    features_dict["StudyInstanceUID"].append(study_id)


feat_df = pd.DataFrame(features_dict)
folds_df = pd.read_csv("../data/train_kfold.csv")
feat_df = feat_df.merge(folds_df, on="StudyInstanceUID")
feat_df.to_csv(f"../data/train_{EXP_NAME}_features_kfold.csv", index=False)