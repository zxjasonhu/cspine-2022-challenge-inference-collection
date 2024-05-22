import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd
import sys
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
sys.path.insert(0, "../../src/")
from skp import builder
from tqdm import tqdm
from utils import create_dir


def rescale(x):
    x = x / x.max()
    x = x - 0.5
    x = x * 2.0
    return x


def add_buffer(x1, x2, max_dist, buff=0.1):
    add_dist = int(buff * (x2 - x1))
    x1, x2 = x1 - add_dist, x2 + add_dist
    x1 = max(0, x1)
    x2 = min(max_dist, x2)
    return x1, x2


CONFIG = "../configs/seg/seg100.yaml"
EXP_NAME = osp.basename(CONFIG).replace(".yaml", "")
SAVE_DIR = f"../../data/train-round2-segmentations-{EXP_NAME}/"
SAVE_LABEL_DIR = f"../../data/label-round2-segmentations-{EXP_NAME}/"
PNG_DIR = "../../data/pngs/"
SEG_LABELS_DIR = "../../data/segmentations-numpy/"

create_dir(SAVE_DIR)
create_dir(SAVE_LABEL_DIR)
config = OmegaConf.load(CONFIG)

checkpoint_paths = np.sort(glob.glob(osp.join("../../experiments/", EXP_NAME,
                                              "sbn", "fold*", "checkpoints", "best.ckpt")))
model_dict = {}
for fold, path in enumerate(checkpoint_paths):
    tmp_config = config.copy()
    tmp_config.model.load_pretrained = str(path)
    model_dict[fold] = builder.build_model(tmp_config).eval().cuda()

df = pd.read_csv("../../data/train_metadata.csv")
folds_df = pd.read_csv("../../data/train_kfold.csv")


studies = glob.glob(osp.join(SEG_LABELS_DIR, "*.npy"))
studies = [osp.basename(_).replace(".npy", "") for _ in studies]
assert len(studies) == 87
df = df[df.StudyInstanceUID.isin(studies)]
folds_df = folds_df[folds_df.StudyInstanceUID.isin(studies)]
folds_dict = {row.StudyInstanceUID: row.outer for _, row in folds_df.iterrows()}

for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    study_files = study_df.filename
    study_files = [_.replace("dcm", "png") for _ in study_files]
    study_stack = np.stack([cv2.imread(osp.join(PNG_DIR, fi), cv2.IMREAD_UNCHANGED) for fi in study_files])
    study_label = np.load(osp.join(SEG_LABELS_DIR, f"{study_id}.npy"))
    assert study_stack.shape == study_label.shape, f"{study_stack.shape} != {study_label.shape}"
    Z, H, W = study_stack.shape
    X = torch.from_numpy(study_stack).float().unsqueeze(0).unsqueeze(0)
    X = F.interpolate(X, size=(192, 192, 192), mode="nearest")
    rescale_factors = [Z / 192, H / 192, W / 192]
    X = rescale(X).cuda()
    with torch.no_grad():
        pseg = model_dict[folds_dict[study_id]](X)
        pseg = torch.sigmoid(pseg).cpu()
        # Calculate probability of C-spine
        p_cervical = pseg.squeeze(0)[:7].sum(0).numpy()
        spine_map = torch.argmax(pseg.squeeze(0)[:7], dim=0) + 1
        spine_map[p_cervical < 0.5] = 0
        spine_map = F.interpolate(spine_map.unsqueeze(0).unsqueeze(0).float(), size=(Z, H, W), mode="nearest").squeeze(0).squeeze(0).numpy()
    coords = np.vstack(np.where(p_cervical >= 0.25))
    coords[0] = coords[0] * rescale_factors[0]
    coords[1] = coords[1] * rescale_factors[1]
    coords[2] = coords[2] * rescale_factors[2]
    coords = coords.astype("int")
    z1, z2, h1, h2, w1, w2 = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max(), coords[2].min(),\
                             coords[2].max()
    z1, z2 = add_buffer(z1, z2, Z, 0.1)
    h1, h2 = add_buffer(h1, h2, H, 0.1)
    w1, w2 = add_buffer(w1, w2, W, 0.1)
    cropped_stack = study_stack[z1:z2+1, h1:h2+1, w1:w2+1]
    print(f"({Z}, {H}, {W}) --> {cropped_stack.shape}")
    cropped_pseg = (spine_map[z1:z2+1, h1:h2+1, w1:w2+1] * 255 / 7).astype("uint8")
    z, h, w = cropped_pseg.shape
    rescale_factors = [192 / z, 192 / h, 192 /w]
    cropped_label = study_label[z1:z2+1, h1:h2+1, w1:w2+1]
    cropped_stack = np.stack([cropped_stack, cropped_pseg], axis=-1)
    cropped_stack = zoom(cropped_stack, rescale_factors + [1], order=0, prefilter=False)
    cropped_label = zoom(cropped_label, rescale_factors, order=0, prefilter=False)
    cropped_label[cropped_label > 7] = 0
    print(np.unique(cropped_label))
    np.save(osp.join(SAVE_DIR, f"{study_id}.npy"), cropped_stack)
    np.save(osp.join(SAVE_LABEL_DIR, f"{study_id}.npy"), cropped_label)







