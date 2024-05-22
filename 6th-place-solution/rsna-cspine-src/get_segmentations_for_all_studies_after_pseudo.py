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

from skp import builder


def predict(model, X):
    X = torch.from_numpy(X).float().cuda().unsqueeze(0).unsqueeze(0)
    return torch.sigmoid(model(X)).cpu()


torch.set_grad_enabled(False)

IMSIZE = 192.
PLOT = False

df = pd.read_csv("../data/train_metadata.csv")

cfg = OmegaConf.load("configs/mk3d/mk3d008.yaml")
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")
checkpoints = np.sort(glob.glob("../experiments/mk3d008/sbn/fold*/checkpoints/last.ckpt"))
models = []
for ckpt in checkpoints:
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models.append(tmp_model)

study_df_list = []

for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    try:
        study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
        study_images = study_df.filename.tolist()
        study_images = [osp.join("../data/pngs", i.replace("dcm", "png")) for i in study_images]
        X = np.stack([cv2.imread(im, 0) for im in study_images])
        z, h, w = X.shape
        rescale_factor = [IMSIZE / z, IMSIZE / h, IMSIZE / w]
        X = zoom(X, rescale_factor, order=0, prefilter=False)
        X = dataset.preprocess(X)
        p = torch.stack([predict(m, X) for m in models]).mean(0)
        p = torch.nn.functional.interpolate(p, size=(z, h, w), mode="nearest").squeeze(0)
        for C in range(7):
            level_present_on_slice = np.asarray([0] * len(study_images))
            seg = np.vstack(np.where(p[C].numpy() >= 0.5))
            z1, z2, h1, h2, w1, w2 = seg[0].min(), seg[0].max(), seg[1].min(), seg[1].max(), seg[2].min(), seg[2].max()
            level_present_on_slice[z1:z2+1] = 1
            study_df[f"C{C+1}"] = level_present_on_slice
            study_df[f"C{C+1}_coords"] = f"{h1}, {h2}, {w1}, {w2}"
        study_df_list.append(study_df)
    except Exception as e:
        print(e)
        print(f"Failed {study_id}")

    if PLOT:
        p_spine = p.sum(0)
        p = torch.argmax(p, dim=0) + 1
        p[p_spine < 0.5] = 0
        p = p.numpy()
        p = p / 8.0
        p *= 255.
        p = p.astype("uint8")
        for i in range(0, p.shape[-1], 10):
            if np.sum(p[..., i]) > 0:
                plt.imshow(p[:,:,i], cmap="gray")
                plt.show()

df = pd.concat(study_df_list)
df.to_csv("../data/train_slices_with_vertebra_levels.csv", index=False)