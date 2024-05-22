import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F

from collections import defaultdict
from monai.inferers import sliding_window_inference
from monai.metrics.utils import get_mask_edges, get_surface_distance
from omegaconf import OmegaConf
from skp import builder
from skp.metrics.segmentation import DSC
from tqdm import tqdm

torch.set_grad_enabled(False)


class Hausdorff:

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.metric_dict = defaultdict(list)

    @staticmethod
    def _compute_hausdorff(pred, gt, max_dist):
        if np.all(pred == gt):
            return 0.0
        (edges_pred, edges_gt) = get_mask_edges(pred, gt)
        surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
        if surface_distance.shape == (0,):
            return 0.0
        dist = surface_distance.max()
        if dist > max_dist:
            return 1.0
        return 1 - dist / max_dist

    def update(self, p, t):
        # p.shape = t.shape = (Z, C, H, W)
        z, c, h, w = p.shape
        for each_class in range(p.shape[1]):
            class_p = (p[:, each_class] >= self.threshold).astype("float")
            # Supposedly, metric scales all dimensions of the image volume to 1, so it becomes a cube
            # All images are squares in axial dimensions, so maximum distance becomes H*sqrt(3)
            hd_ab = self._compute_hausdorff(class_p, t[:, each_class], max_dist=h*np.sqrt(3))
            hd_ba = self._compute_hausdorff(t[:, each_class], class_p, max_dist=h*np.sqrt(3))
            self.metric_dict[f"hd_ab_{each_class:02d}"].append(hd_ab)
            self.metric_dict[f"hd_ba_{each_class:02d}"].append(hd_ba)

    def compute(self):
        for k, v in self.metric_dict.items():
            self.metric_dict[k] = np.mean(v)
        self.metric_dict["hd_ab_mean"] = np.mean([v for k, v in self.metric_dict.items() if "ab" in k])
        self.metric_dict["hd_ba_mean"] = np.mean([v for k, v in self.metric_dict.items() if "ba" in k])
        self.metric_dict["hd_mean"] = (self.metric_dict["hd_ab_mean"] + self.metric_dict["hd_ba_mean"]) / 2.0
        return self.metric_dict


def load_image_volume(folder, image_size, cv2_flag=cv2.IMREAD_UNCHANGED, twodc=True, return_original=True):
    images = np.sort(glob.glob(osp.join(folder, "*")))
    x = [cv2.imread(im, cv2_flag) for im in images]
    if x[0].ndim == 2:
        x = [np.expand_dims(im, axis=-1) for im in x]
    x = np.stack(x)
    # x.shape = (Z, H, W, C)
    dims = x.shape[:3]
    if return_original:
        x_orig = x.copy()
    if twodc:
        channel1 = np.concatenate([np.expand_dims(x[0], axis=0), x[:-1]])
        channel3 = np.concatenate([x[1:], np.expand_dims(x[-1], axis=0)])
        x = np.concatenate([channel1, x, channel3], axis=-1)
    if image_size:
        x = [cv2.resize(im, tuple(image_size)) for im in x]
    if return_original:
        return np.ascontiguousarray(x), x_orig, dims
    return np.ascontiguousarray(x), dims


def load_model(config, checkpoint, device="cuda:0"):
    cfg = OmegaConf.load(config)
    cfg.model.load_pretrained = str(checkpoint)
    model = builder.build_model(cfg)
    return model.eval().to(device)




def load_preprocessor(config):
    cfg = OmegaConf.load(config)
    return builder.build_dataset(cfg, data_info={"inputs": 0, "labels": 0}, mode="predict")


def predict(input_2d,
            input_3d,
            classifier,
            segment_2d,
            segment_3d,
            segment_25d,
            num_images,
            original_imsize,
            roi_size=None,
            class_threshold=0.025,
            seg_threshold=None):
    # input_2d.shape = (Z, C, H, W)
    # input_3d.shape = (1, C, Z, H, W)
    torch.set_grad_enabled(False)
    if classifier:
        # Assumes that the classifier uses the same input as the 2D segmentation model
        # In our case, the plan is to use 512 x 512 2Dc images for both
        class_preds = classifier(input_2d)[:, 3]
        assert len(class_preds) == num_images
        class_preds = torch.sigmoid(class_preds).cpu()
        positives = torch.where(class_preds >= class_threshold)[0]
        start, stop = positives[0], positives[-1]
        start, stop = max(0, start - 5), min(num_images, stop + 5)
    else:
        start, stop = 0, num_images
    if segment_2d:
        input_2d = input_2d[start:stop + 1]
        seg2d_preds = torch.sigmoid(torch.cat([segment_2d(input_2d[i:i+16]) for i in range(0, len(input_2d), 16)]))
        if seg2d_preds.size(2) != original_imsize[0] or seg2d_preds.size(3) != original_imsize[1]:
            seg2d_preds = F.interpolate(seg2d_preds, size=original_imsize)
        seg2d_preds = seg2d_preds[:, :3].cpu()
    if segment_3d:
        input_3d = input_3d[:, :, start:stop + 1]
        seg3d_preds = torch.sigmoid(sliding_window_inference(inputs=input_3d, roi_size=roi_size, sw_batch_size=8,
                                               predictor=segment_3d).squeeze(0).swapaxes(0, 1))
        if seg3d_preds.size(2) != original_imsize[0] or seg3d_preds.size(3) != original_imsize[1]:
            seg3d_preds = F.interpolate(seg3d_preds, size=original_imsize)
        seg3d_preds = seg3d_preds[:, :3].cpu()
    if segment_25d:
        if not segment_2d: input_2d = input_2d[start:stop+1]
        input_2d = input_2d[:,[2,1,0]] # Need to reverse the channels because of my stupid code
        seg25d_preds = torch.sigmoid(torch.cat([segment_25d(input_2d[i:i+16]) for i in range(0, len(input_2d), 16)]))
        # predictions will be of shape (Z, 9, H, W)
        # top = seg25d_preds[:, :3]
        seg25d_preds = seg25d_preds[:, 3:6]
        # bot = seg25d_preds[:, 6:]
        # seg25d_preds = torch.cat([top[0].unsqueeze(0), top[:-1]]) + mid + \
        #     torch.cat([bot[1:], bot[-1].unsqueeze(0)])
        # seg25d_preds /= 3
        if seg25d_preds.size(2) != original_imsize[0] or seg25d_preds.size(3) != original_imsize[1]:
            seg25d_preds = F.interpolate(seg25d_preds, size=original_imsize).cpu()
    segmentation_list = []
    if segment_2d:
        segmentation_list.append(seg2d_preds)
    if segment_3d:
        segmentation_list.append(seg3d_preds)
    if segment_25d:
        segmentation_list.append(seg25d_preds)
    segmentation_preds = torch.stack(segmentation_list).mean(0)
    segmentation_output = torch.zeros((num_images, 3, original_imsize[0], original_imsize[1]))
    segmentation_output[start:stop+1] = segmentation_preds
    if isinstance(seg_threshold, float):
        segmentation_output = (segmentation_output >= seg_threshold).float()
    return segmentation_output


CONFIGS = {
    "classifier": "configs/cls/cls003.yaml",
    "segment_2d": "configs/mks/mk013.yaml",
    "segment_3d": "configs/mk3d/mk3d006.yaml",
    "segment_25d": "configs/mks/mk015.yaml"
}

CHECKPOINTS = {}
for k, v in CONFIGS.items():
    best_checkpoints = glob.glob(osp.join("../experiments",
                                          v.split("/")[-1].replace(".yaml", ""),
                                          "sbn", "fold*", "checkpoints", "best.ckpt"))
    CHECKPOINTS[k] = np.sort(best_checkpoints)

df = pd.read_csv("../data/train_3d_kfold.csv")
df["filename"] = df.filename.apply(lambda x: osp.join("../data", x))
df["label"] = df.label.apply(lambda x: osp.join("../data", x))

# All models use the same preprocessor
preprocess = load_preprocessor(CONFIGS["classifier"]).preprocess

for kfold in range(5):
    test_df = df[df.outer == kfold]
    classifier = load_model(CONFIGS["classifier"], CHECKPOINTS["classifier"][kfold])
    segment_2d = load_model(CONFIGS["segment_2d"], CHECKPOINTS["segment_2d"][kfold])
    segment_3d = None#load_model(CONFIGS["segment_3d"], CHECKPOINTS["segment_3d"][kfold])
    segment_25d = None#load_model(CONFIGS["segment_25d"], CHECKPOINTS["segment_25d"][kfold])
    dsc_metric = DSC(apply_sigmoid=False)
    hd_metric = Hausdorff()
    for study, gt in tqdm(zip(test_df.filename, test_df.label), total=len(test_df)):
        X_2Dc, X_orig, dims = load_image_volume(study, (512, 512), twodc=True, return_original=True)
        X_2Dc = preprocess(X_2Dc).transpose(0, 3, 1, 2)
        X_orig = preprocess(X_orig).transpose(3, 0, 1, 2)
        X_2Dc = torch.from_numpy(X_2Dc).cuda().float()
        X_orig = torch.from_numpy(X_orig).unsqueeze(0).cuda().float()
        y = predict(input_2d=X_2Dc, input_3d=X_orig,
                    classifier=classifier, segment_2d=segment_2d, segment_3d=segment_3d,
                    segment_25d=segment_25d,
                    num_images=dims[0], original_imsize=dims[1:], roi_size=(64, 224, 224))
        ground_truth, _ = load_image_volume(gt, image_size=None, twodc=False, return_original=False)
        ground_truth = ground_truth.transpose(0, 3, 1, 2)
        ground_truth = torch.from_numpy(ground_truth).float()
        dsc_metric.update(y, ground_truth)
        hd_metric.update(y.numpy(), ground_truth.numpy())

    print(dsc_metric.compute())
    print(hd_metric.compute())