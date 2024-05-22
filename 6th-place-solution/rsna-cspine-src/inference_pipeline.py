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


def create_dir(d): 
    if not osp.exists(d): os.makedirs(d)


CONFIG     = "configs/mks/mk006.yaml"
CHECKPOINT = {
    0 : '../experiments/mk006/sbn/fold0/checkpoints/epoch=019-vm=0.9062.ckpt',
    1 : '../experiments/mk006/sbn/fold1/checkpoints/epoch=019-vm=0.9083.ckpt',
    2 : '../experiments/mk006/sbn/fold2/checkpoints/epoch=018-vm=0.9119.ckpt',
    3 : '../experiments/mk006/sbn/fold3/checkpoints/epoch=018-vm=0.9100.ckpt',
    4 : '../experiments/mk006/sbn/fold4/checkpoints/epoch=018-vm=0.9145.ckpt',
}
CSVFILE    = "../data/train_3d_kfold.csv"
DATADIR    = "../data/"
IMAGE_SIZE = [512, 512]
DEVICE     = "cuda:0"
SEGDIR     = "../data/predicted-segmentations/"

create_dir(SEGDIR)

cfg = OmegaConf.load(CONFIG)
cfg.model.params.encoder_params.pretrained = False 
df = pd.read_csv(CSVFILE)
preprocessor = Preprocessor(image_range=[0,255], input_range=[0,1], mean=[0.5]*3, sdev=[0.5]*3)

for KFOLD in range(5):
    cfg.model.load_pretrained = CHECKPOINT[KFOLD]
    model = builder.build_model(cfg)
    model = model.to(DEVICE).eval()
    test_df = df[df.outer == KFOLD]
    metric = DSC()
    hd1 = defaultdict(list)
    hd2 = defaultdict(list)
    for scan, seg in tqdm(zip(test_df.filename, test_df.label), total=len(test_df)):
        slices = np.sort(glob.glob(osp.join(DATADIR, scan, "*.png")))
        labels = np.sort(glob.glob(osp.join(DATADIR, seg, "*.png")))
        X = np.asarray([
                np.expand_dims(
                    cv2.resize(
                        cv2.imread(each_slice, 0), 
                        tuple(IMAGE_SIZE)
                    ),
                    axis=-1
                )
                for each_slice in slices
        ])
        y = np.asarray([
                cv2.resize(
                    cv2.imread(each_seg), 
                    tuple(IMAGE_SIZE)
                )
            for each_seg in labels
        ])
        X = preprocessor(X)
        X, y = X.transpose(0, 3, 1, 2), y.transpose(0, 3, 1, 2)
        X, y = np.ascontiguousarray(X), np.ascontiguousarray(y)
        with torch.no_grad():
            X, y = torch.from_numpy(X).float().to(DEVICE), torch.from_numpy(y).float().to(DEVICE)
            p = torch.cat([
                model(X[i:i+16])
                for i in range(0, len(X), 16)
            ], dim=0)
            metric.update(p, y)
        # p.shape = (Z, C, H, W)
        p = torch.sigmoid(p).cpu().numpy()
        p_binarize = (p >= 0.5).astype("float")
        y = y.cpu().numpy()
        for each_class in range(p.shape[1]):
            hd1[each_class].append(1 - compute_hausdorff(p_binarize[:,each_class], y[:,each_class], max_dist=IMAGE_SIZE[0] * np.sqrt(3)))
            hd2[each_class].append(1 - compute_hausdorff(y[:,each_class], p_binarize[:,each_class], max_dist=IMAGE_SIZE[0] * np.sqrt(3)))
        p = (p * 255.0).astype("uint8")
        case_segdir = "/".join(slices[0].split("/")[3:-1])
        create_dir(osp.join(SEGDIR, case_segdir))
        for i in range(p.shape[0]):
            filepath = osp.join(SEGDIR, case_segdir, slices[i].split("/")[-1])
            _ = cv2.imwrite(filepath, p[i].transpose(1, 2, 0))
    results = metric.compute()
    print(f"DSC [ignore empty] : {results['dsc_ignore_mean']:0.4f}")
    print(f"DSC [empty is one] : {results['dsc_empty1_mean']:0.4f}")
    print(f"HD  [a,b]          : {np.mean([np.mean(v) for k,v in hd1.items()]):0.4f}")
    print(f"HD  [b,a]          : {np.mean([np.mean(v) for k,v in hd2.items()]):0.4f}")
