import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd
import torch

from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from skp import builder
from etl.utils import create_dir


def get_cam(model, x, labels=["fracture"]):
    with torch.no_grad():
        x = model.backbone(x)
        x = x.squeeze(0)
        linear_weights = model.classifier.weight
        cam_dict = {}
        for class_idx, class_w in enumerate(linear_weights):
            cam_dict[labels[class_idx]] = torch.zeros((x.size(-3), x.size(-2), x.size(-1))).to(x.device)
            for idx, w in enumerate(class_w):
                cam_dict[labels[class_idx]] += w * x[idx]
    return {k: v.cpu().numpy() for k, v in cam_dict.items()}


def rescale(x):
    x = x / x.max()
    x = x - 0.5
    x = x * 2.0
    return x


torch.set_grad_enabled(False)

CHUNK_DIR = "../data/train-individual-vertebrae/"
SAVE_IMAGES_DIR = "../data/train-cropped-vertebra/"
create_dir(SAVE_IMAGES_DIR)

cfg = OmegaConf.load("configs/chunk/chunk200.yaml")
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")

checkpoints = {
    f"fold{i}": f"../experiments/chunk200/sbn/fold{i}/checkpoints/best.ckpt" for i in range(3)
}

models_dict = {}
for fold, ckpt in checkpoints.items():
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models_dict[fold] = tmp_model


train_df = pd.read_csv("../data/train.csv")
fractured_study_vertebra = []
for level in [f"C{i + 1}" for i in range(7)]:
    tmp_df = train_df[train_df[level] == 1]
    study_level_names = [f"{study_id}_{level}" for study_id in tmp_df.StudyInstanceUID]
    fractured_study_vertebra.extend(study_level_names)

all_vertebra = np.sort(glob.glob(osp.join(CHUNK_DIR, "*.npy")))
all_filenames = []
all_cas = []
for each_vertebra in tqdm(all_vertebra):
    chunk = np.load(each_vertebra)
    filenames = [osp.join(SAVE_IMAGES_DIR, f"{osp.basename(each_vertebra.replace('.npy', ''))}_{ind:03d}.png")
                 for ind in range(len(chunk))]
    all_filenames.extend([osp.basename(fi) for fi in filenames])
    for each_file, each_slice in zip(filenames, chunk):
        _ = cv2.imwrite(each_file, each_slice)
    if osp.basename(each_vertebra).replace(".npy", "") not in fractured_study_vertebra:
        # NEGATIVE
        all_cas.extend([0] * len(filenames))
        continue
    chunk_torch = rescale(torch.from_numpy(chunk).float()).unsqueeze(0).unsqueeze(0)
    chunk_torch = torch.nn.functional.interpolate(chunk_torch, size=(64, 288, 288), mode="trilinear").cuda()
    cam_list = []
    for fold, model in models_dict.items():
        cam = get_cam(model, chunk_torch.cuda())["fracture"]
        cam_list.append(cam)
    cam = np.stack(cam_list).mean(0)
    # Reshape and take the max to get the CAS
    cas = cam.reshape(cam.shape[0], -1).max(-1)
    # Scale to [0, 1]
    cas[cas < 0] = 0
    cas /= np.max(cas)
    cas = zoom(cas, chunk.shape[0] / float(len(cas)), order=0, prefilter=False)
    all_cas.extend(list(cas))


df = pd.DataFrame({"filename": all_filenames, "fracture_cas": all_cas})
df["fracture_cas_binary"] = (df.fracture_cas > 0.2).astype("int")
print(df.fracture_cas_binary.value_counts())
folds_df = pd.read_csv("../data/train_kfold.csv")
df["StudyInstanceUID"] = df.filename.apply(lambda x: osp.basename(x).split("_")[0])
df = df.merge(folds_df, on="StudyInstanceUID")
df.to_csv("../data/train_quick_cas_kfold.csv", index=False)


import pandas as pd


x = pd.read_csv("../data/train.csv")
y = pd.read_csv("../data/train_quick_cas_kfold.csv")

# Only take negatives from negative studies
negatives = x[x.patient_overall == 0].StudyInstanceUID.tolist()
pos_y = y[y.fracture_cas_binary == 1]
neg_y = y[y.StudyInstanceUID.isin(negatives)]

y = pd.concat([pos_y, neg_y])
print(y.fracture_cas_binary.value_counts())

y.to_csv("../data/train_quick_cas_kfold_subset.csv", index=False)