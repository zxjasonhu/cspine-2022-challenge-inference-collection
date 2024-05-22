import cv2
import glob
import numpy as np
import os, os.path as osp
import pandas as pd
import torch

from omegaconf import OmegaConf

from skp import builder
from skp.data.transforms import Preprocessor
from tqdm import tqdm


def create_dir(d):
    if not osp.exists(d): os.makedirs(d)


def load_image_volume(image_paths, image_size, cv2_flag=None, twodc=False):
    if not cv2_flag: cv2_flag = cv2.IMREAD_UNCHANGED
    x = [cv2.imread(im, cv2_flag) for im in image_paths]
    if image_size: x = [cv2.resize(im, tuple(image_size)) for im in x]
    if x[0].ndim == 2: x = [np.expand_dims(im, axis=-1) for im in x]
    x = np.stack(x)
    # x.shape = (Z, H, W, C)
    if twodc:
        channel1 = np.concatenate([np.expand_dims(x[0], axis=0), x[:-1]])
        channel3 = np.concatenate([x[1:], np.expand_dims(x[-1], axis=0)])
        x = np.concatenate([channel1, x, channel3], axis=-1)
    return np.ascontiguousarray(x)


KFOLD      = [4]
CONFIG     = "configs/mkpos/mkpos003.yaml"
CHECKPOINT = [
    "../experiments/mkpos003/sbn/fold0/checkpoints/best.ckpt",
    "../experiments/mkpos003/sbn/fold1/checkpoints/best.ckpt",
    "../experiments/mkpos003/sbn/fold2/checkpoints/best.ckpt",
    "../experiments/mkpos003/sbn/fold3/checkpoints/best.ckpt",
    "../experiments/mkpos003/sbn/fold4/checkpoints/best.ckpt",
]

CSVFILE    = "../data/train_positive_slices.csv"
DATADIR    = "../data/"
IMAGE_SIZE = [384, 384]
DEVICE     = "cuda:0"
SAVE_XDIR = "../data/train_numpy_chunks_cropped/"
SAVE_YDIR = "../data/label_numpy_chunks_cropped/"

create_dir(SAVE_XDIR)
create_dir(SAVE_YDIR)
cfg = OmegaConf.load(CONFIG)
cfg.model.params.encoder_params.pretrained = False 
df = pd.read_csv(CSVFILE)

image_sizes = {}
for fold in KFOLD:
    cfg.model.load_pretrained = CHECKPOINT[fold]
    model = builder.build_model(cfg)
    model = model.to(DEVICE).eval()
    test_df = df[df.outer == fold]
    preprocessor = builder.build_dataset(cfg, data_info={"inputs": 0, "labels": 0}, mode="predict").preprocess
    for case_day, _df in tqdm(test_df.groupby("case_day"), total=len(test_df.case_day.unique())):
        _df = _df.sort_values("slice_id", ascending=True)
        X = load_image_volume([osp.join(DATADIR, _) for _ in _df.filename], IMAGE_SIZE, twodc=True)
        y = load_image_volume([osp.join(DATADIR, _) for _ in _df.label], IMAGE_SIZE, twodc=False)
        X_flat = load_image_volume([osp.join(DATADIR, _) for _ in _df.filename], IMAGE_SIZE, twodc=False)
        X = preprocessor(X)
        # X.shape = (Z, H, W, C)
        X = X.transpose(0, 3, 1, 2)
        # X.shape = (Z, C, H, W)
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(DEVICE)
            p = torch.cat([model(X[i:i+16]) for i in range(0, len(X), 16)])
        p = torch.sigmoid(p).cpu().numpy().transpose(0, 2, 3, 1)
        CHUNK_SIZE = 32
        OVERLAP = 0.125
        if p.shape[0] < 32:
            top = (32 - p.shape[0]) // 2
            bot = (32 - p.shape[0]) - top
            filler_top = np.stack([np.zeros_like(p[0])] * top)
            filler_bot = np.stack([np.zeros_like(p[0])] * bot)
            p = np.concatenate([filler_top, p, filler_bot])
            filler_top = np.stack([np.zeros_like(X_flat[0])] * top)
            filler_bot = np.stack([np.zeros_like(X_flat[0])] * bot)
            X_flat = np.concatenate([filler_top, X_flat, filler_bot])
            filler_top = np.stack([np.zeros_like(y[0])] * top)
            filler_bot = np.stack([np.zeros_like(y[0])] * bot)
            y = np.concatenate([filler_top, y, filler_bot])
            assert p.shape[0] == 32
        p = (p * 255).astype("uint8")
        X_flat = np.concatenate([X_flat, p], axis=-1)
        assert X_flat.shape[:3] == y.shape[:3]
        idx0 = np.arange(0, len(X_flat) - CHUNK_SIZE, int(CHUNK_SIZE * OVERLAP))
        idx1 = np.arange(len(X_flat) - CHUNK_SIZE, len(X_flat), CHUNK_SIZE)
        idx = np.concatenate([idx0, idx1])
        for chunk_idx, chunk in enumerate(idx): 
            np.save(osp.join(SAVE_XDIR, f"{case_day}_chunk{chunk_idx:02d}.npy"), X_flat[chunk:chunk+CHUNK_SIZE])
            np.save(osp.join(SAVE_YDIR, f"{case_day}_chunk{chunk_idx:02d}.npy"), y[chunk:chunk+CHUNK_SIZE])


df = pd.read_csv("../data/train_kfold.csv")
fold_cols = [c for c in df.columns if "inner" in c or "outer" in c]
df = df[["case_day"] + fold_cols]
df = df.drop_duplicates().reset_index(drop=True)
chunks = glob.glob(osp.join(SAVE_XDIR, "*.npy"))
chunks = [_.replace("../data/", "") for _ in chunks]
chunk_df = pd.DataFrame(dict(filename=chunks))
chunk_df["label"] = chunk_df.filename.apply(lambda x: x.replace("train", "label"))
chunk_df["case_day"] = chunk_df.filename.apply(lambda x: x.split("/")[-1].split("_")[0])
df = df.merge(chunk_df, on="case_day")
df.to_csv("../data/train_3d_chunks_with_seg_pos_only_kfold.csv", index=False)