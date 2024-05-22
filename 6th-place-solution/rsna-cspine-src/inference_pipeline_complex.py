import cv2
import glob
import numpy as np
import os, os.path as osp
import pandas as pd
import torch

from collections import defaultdict
from monai.inferers import sliding_window_inference
from monai.metrics.utils import get_mask_edges, get_surface_distance
from omegaconf import OmegaConf

from skp import builder
from skp.data.transforms import Preprocessor
from skp.metrics.segmentation import DSC
from tqdm import tqdm


def compute_hausdorff(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()
    if dist > max_dist:
        return 1.0
    return dist / max_dist


def load_image_volume(folder, image_size, cv2_flag=None, twodc=False):
    images = np.sort(glob.glob(osp.join(folder, "*")))
    if not cv2_flag: cv2_flag = cv2.IMREAD_UNCHANGED
    x = [cv2.imread(im, cv2_flag) for im in images]
    if image_size: x = [cv2.resize(im, tuple(image_size)) for im in x]
    if x[0].ndim == 2: x = [np.expand_dims(im, axis=-1) for im in x]
    x = np.stack(x)
    # x.shape = (Z, H, W, C)
    if twodc:
        channel1 = np.concatenate([np.expand_dims(x[0], axis=0), x[:-1]])
        channel3 = np.concatenate([x[1:], np.expand_dims(x[-1], axis=0)])
        x = np.concatenate([channel1, x, channel3], axis=-1)
    return np.ascontiguousarray(x)


KFOLD      = [0, 1, 2, 3, 4]
CONFIG     = "configs/mks/mk008.yaml"
CHECKPOINT = [
    "../experiments/mk008/sbn/fold0/checkpoints/best.ckpt",
    "../experiments/mk008/sbn/fold1/checkpoints/best.ckpt",
    "../experiments/mk008/sbn/fold2/checkpoints/best.ckpt",
    "../experiments/mk008/sbn/fold3/checkpoints/best.ckpt",
    "../experiments/mk008/sbn/fold4/checkpoints/best.ckpt",
]

ADDITIONAL_CONFIG = "configs/mkpos/mkpos001.yaml"
ADDITIONAL_CHECKPOINTS = [
    "../experiments/mkpos001/sbn/fold0/checkpoints/best.ckpt",
    "../experiments/mkpos001/sbn/fold1/checkpoints/best.ckpt",
    "../experiments/mkpos001/sbn/fold2/checkpoints/best.ckpt",
    "../experiments/mkpos001/sbn/fold3/checkpoints/best.ckpt",
    "../experiments/mkpos001/sbn/fold4/checkpoints/best.ckpt",
]
CSVFILE    = "../data/train_3d_kfold.csv"
DATADIR    = "../data/"
ROI_SIZE   = [64, 192, 192] # only used for 3D sliding window inference
IMAGE_SIZE = [512, 512]
DEVICE     = "cuda:0"
SEG_DIR    = None
MODEL_DIM  = "2Dx2"

cfg = OmegaConf.load(CONFIG)
cfg.model.params.encoder_params.pretrained = False 
df = pd.read_csv(CSVFILE)
if len(ADDITIONAL_CHECKPOINTS) > 0:
    assert ADDITIONAL_CONFIG != ""
    assert MODEL_DIM == "2Dx2"
    supp_cfg = OmegaConf.load(ADDITIONAL_CONFIG)
    supp_cfg.model.params.encoder_params.pretrained = False

fold_results_dict = defaultdict(list)
for fold in KFOLD:
    cfg.model.load_pretrained = CHECKPOINT[fold]
    model = builder.build_model(cfg)
    model = model.to(DEVICE).eval()
    if len(ADDITIONAL_CHECKPOINTS) > 0:
        supp_cfg.model.load_pretrained = ADDITIONAL_CHECKPOINTS[fold]
        supp_model = builder.build_model(supp_cfg)
        supp_model = supp_model.to(DEVICE).eval()
    test_df = df[df.outer == fold]
    preprocessor = builder.build_dataset(cfg, data_info={"inputs": 0, "labels": 0}, mode="predict").preprocess
    metric = DSC(apply_sigmoid=False)
    hd1, hd2 = defaultdict(list), defaultdict(list)
    for scan, seg in tqdm(zip(test_df.filename, test_df.label), total=len(test_df)):
        X = load_image_volume(osp.join(DATADIR, scan), IMAGE_SIZE, twodc=True)
        y = load_image_volume(osp.join(DATADIR, seg) , IMAGE_SIZE)
        if SEG_DIR:
            S = load_image_volume(osp.join(DATADIR, seg.replace("segmentations", SEG_DIR)), IMAGE_SIZE)
            X = np.concatenate([X, S], axis=-1)
        X = preprocessor(X)
        X, y = X.transpose(3, 0, 1, 2), y.transpose(3, 0, 1, 2)
        # X.shape = (C, Z, H, W) ; y.shape = (C, Z, H, W)
        with torch.no_grad():
            X, y = torch.from_numpy(X).float().to(DEVICE), torch.from_numpy(y).float().to(DEVICE)
            if MODEL_DIM == "3D":
                X = X.unsqueeze(0)
                # X.shape = (1, C, Z, H, W)
                p = sliding_window_inference(inputs=X, roi_size=ROI_SIZE, sw_batch_size=8, predictor=model)
                p = p.squeeze(0).swapaxes(0, 1)
                p = torch.sigmoid(p)
                # p.shape = (Z, C, H, W)
                metric.update(p[:,:3], y.swapaxes(0, 1))
            elif MODEL_DIM == "2D":
                X  = X.swapaxes(0, 1)
                p = torch.cat([model(X[i:i+16]) for i in range(0, len(X), 16)])
                p = torch.sigmoid(p)
                # p.shape (Z, C, H, W)
                metric.update(p[:,:3], y.swapaxes(0, 1))
            elif MODEL_DIM == "2Dx2":
                # This is used when you are ensembling a model trained on all images
                # and a model trained on positive images 
                # First, perform regular inference with the model trained on all images
                X = X.swapaxes(0, 1)
                p = torch.cat([model(X[i:i+16]) for i in range(0, len(X), 16)])
                # Apply sigmoid
                p = torch.sigmoid(p)
                # This model should also produce a 4th channel with foreground segmentation
                # Use this channel to determine whether an image contains foreground
                fg = p[:,3].amax((-1, -2))
                fg_indices = torch.where(fg >= 0.5)[0].cpu().numpy()
                # Now perform inference with other model only on these slices
                supp_p = torch.cat([supp_model(X[i].unsqueeze(0)) for i in fg_indices])
                supp_p = torch.sigmoid(supp_p)
                # Now only ensemble these slices 
                # Get rid of last channel in initial prediction
                p = p[:,:3]
                for idx, fg_idx in enumerate(fg_indices):
                    p[fg_idx] = torch.stack([p[fg_idx], supp_p[idx]]).mean(0)
                metric.update(p, y.swapaxes(0, 1))
        # Compute Hausdorff
        p = p.swapaxes(0, 1).cpu().numpy()
        p = (p >= 0.5).astype("float")
        y = y.cpu().numpy()
        for each_class in range(3):
            hd1[each_class].append(1 - compute_hausdorff(p[each_class], y[each_class], max_dist=IMAGE_SIZE[0] * np.sqrt(3)))
            hd2[each_class].append(1 - compute_hausdorff(y[each_class], p[each_class], max_dist=IMAGE_SIZE[0] * np.sqrt(3)))
    dsc_results = metric.compute()
    fold_results_dict[fold].append((dsc_results, hd1, hd2))
    print(f"FOLD {fold} RESULTS\n======")
    print(f"DSC[0]  : {dsc_results['dsc_ignore_00']:0.4f}")
    print(f"DSC[1]  : {dsc_results['dsc_ignore_01']:0.4f}")
    print(f"DSC[2]  : {dsc_results['dsc_ignore_02']:0.4f}")
    print(f"DSC     : {dsc_results['dsc_ignore_mean']:0.4f}")
    print(f"HDab[0] : {np.mean(hd1[0]):0.4f}")
    print(f"HDab[1] : {np.mean(hd1[1]):0.4f}")
    print(f"HDab[2] : {np.mean(hd1[2]):0.4f}")
    print(f"HDab    : {np.mean(np.concatenate([v for v in hd1.values()])):0.4f}")
    print(f"HDba[0] : {np.mean(hd2[0]):0.4f}")
    print(f"HDba[1] : {np.mean(hd2[1]):0.4f}")
    print(f"HDba[2] : {np.mean(hd2[2]):0.4f}")
    print(f"HDba    : {np.mean(np.concatenate([v for v in hd2.values()])):0.4f}")

print(f"DSC KFOLD : {np.mean([v[0][0]['dsc_ignore_mean'].item() for v in fold_results_dict.values()]):0.4f}")
print(f"HDab KFOLD : {np.mean([np.mean(list(v[0][1].values())) for v in fold_results_dict.values()]):0.4f}")
