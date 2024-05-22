import os
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List

import albumentations
import cv2
import numpy as np
import pydicom
import torch
from albumentations import ReplayCompose
from skimage import measure
from torch import Tensor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "", "src")
sys.path.insert(0, MODEL_PATH)
from zoo import ClassifierResNet3dCSN2P1D
from zoo import ResNet3dCSN2P1D

configs = {
    "classifier_csn_ir152": {
        "network": ClassifierResNet3dCSN2P1D,
        "encoder_params": {
            "encoder": "r152ir",
            "num_classes": 8,
            "pool": "max"
        }
    },
    "segmentor_csn_ir152": {
        "network": ResNet3dCSN2P1D,
        "encoder_params": {
            "encoder": "r50ir"
        }
    },
}


@dataclass
class BatchSlice:
    i_from: int
    i_to: int
    i_start: int


def get_slices(batch: Tensor, dim=1, window: int = 16, overlap: int = 8) -> List[BatchSlice]:
    num_imgs = batch.size(dim)
    if num_imgs <= window:
        return [BatchSlice(0, num_imgs, 0)]
    stride = window - overlap
    result = []
    current_idx = 0
    while True:
        next_idx = current_idx + window

        if next_idx >= num_imgs:
            current_idx = num_imgs - window
            offset = overlap // 2 if current_idx > 0 else 0
            next_idx = num_imgs
            result.append(BatchSlice(current_idx, next_idx, offset))
            break
        else:
            offset = overlap // 2 if current_idx > 0 else 0
            result.append(BatchSlice(current_idx, next_idx, offset))
        current_idx += stride
    return result


def load_checkpoint(model, checkpoint_path, strict=False, verbose=True):
    if verbose:
        print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {re.sub("^module.", "", k): w for k, w in state_dict.items()}
        orig_state_dict = model.state_dict()
        mismatched_keys = []
        for k, v in state_dict.items():
            ori_size = orig_state_dict[k].size() if k in orig_state_dict else None
            if v.size() != ori_size:
                if verbose:
                    print("SKIPPING!!! Shape of {} changed from {} to {}".format(k, v.size(), ori_size))
                mismatched_keys.append(k)
        for k in mismatched_keys:
            del state_dict[k]
        model.load_state_dict(state_dict, strict=strict)
        del state_dict
        del orig_state_dict
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))
    else:
        model.load_state_dict(checkpoint)
    del checkpoint


def load_model(conf: Dict, checkpoint: str):
    model = conf["network"](**conf["encoder_params"])
    load_checkpoint(model, checkpoint)
    return model.eval()


crop_augs = albumentations.ReplayCompose([
    albumentations.LongestMaxSize(256),
    albumentations.PadIfNeeded(256, 256, border_mode=cv2.BORDER_CONSTANT),
])


class MDAIModel:
    def __init__(self):
        root_path = os.path.dirname(os.path.dirname(__file__))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        cls_models = [
            load_model(configs["classifier_csn_ir152"],
                       os.path.join(root_path, "4th-place", "checkpoints", "swa_3_best_ClassifierResNet3dCSN2P1D_r152ir_0.pth")),
            load_model(configs["classifier_csn_ir152"],
                       os.path.join(root_path, "4th-place", "checkpoints", "swa_5_best_ClassifierResNet3dCSN2P1D_r152ir_1.pth")),
            load_model(configs["classifier_csn_ir152"],
                       os.path.join(root_path, "4th-place", "checkpoints", "swa_5_best_ClassifierResNet3dCSN2P1D_r152ir_2.pth")),
            load_model(configs["classifier_csn_ir152"],
                       os.path.join(root_path, "4th-place", "checkpoints", "swa_5_best_ClassifierResNet3dCSN2P1D_r152ir_3.pth")),
        ]
        self.cls_models = [model.to(self.device) for model in cls_models]
        self.seg_model = load_model(configs["segmentor_csn_ir152"],
                                    os.path.join(root_path, "4th-place", "checkpoints", "256_ResNet3dCSN2P1D_r50ir_1_dice")).to(
            self.device)

    def _combine_scan(self, data, size=512, fix_monochrome: bool = True) -> np.ndarray:
        images = []
        first = None
        last = None
        dicoms = []

        # TODO: implement reading full study from data, this was just an example for local devbox without md.ai
        scan_dir = data["scan_dir"]

        for file in os.listdir(scan_dir):
            if file.endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(scan_dir, file))
                dicoms.append(ds)

        dicoms.sort(key=lambda x: int(x.InstanceNumber))

        #  ::2 is important,
        #  as models were trained with stride 2, as augmentation one can use two predicts with 1::2 and  ::2 for an ensemble
        for ds in dicoms[::2]:
            # todo: here will be something like ds = pydicom.dcmread(BytesIO(file["content"]))
            if not first:
                first = ds
            last = ds

            data = ds.pixel_array
            data = cv2.resize(data, (size, size))
            if fix_monochrome and ds.PhotometricInterpretation == "MONOCHROME1":
                data = np.amax(data) - data
            images.append(data)

        if first and last:
            if last.ImagePositionPatient[2] > first.ImagePositionPatient[2]:
                images = images[::-1]
        return np.array(images)

    def _get_image_cube_segmentation(self, data) -> Dict:
        image_cube = self._combine_scan(data, size=256)
        image_mean = image_cube.mean()
        image_std = image_cube.std()
        h = image_cube.shape[0]

        images = image_cube
        if h % 32 > 0:
            tmp = np.zeros(((h // 32 + 1) * 32, 256, 256))
            tmp[:h] = images
            images = tmp
        images = (images - image_mean) / image_std
        images = np.expand_dims(images, 0)
        sample = {}
        sample['image'] = torch.from_numpy(images).float()
        sample['h'] = h
        return sample

    def _get_image_cubes_classification(self, data, mask_cube: np.ndarray) -> torch.Tensor:

        image_cube = self._combine_scan(data, size=512)
        boxes = {}
        for rprop in measure.regionprops(mask_cube):
            boxes[rprop.label] = rprop.bbox, rprop.area

        image_mean = image_cube.mean()
        image_std = image_cube.std()
        slice_size = 40
        all_images = []
        for li in range(1, 8):
            if li not in boxes:
                all_images.append(np.zeros((3, slice_size, 256, 256)))
            else:
                bbox, area = boxes[li]
                z1, z2 = bbox[0], bbox[3]
                y1, y2 = max(bbox[1] - 16, 0), min(bbox[4] + 16, 256)
                x1, x2 = max(bbox[2] - 16, 0), min(bbox[5] + 16, 256)
                if z2 - z1 < slice_size:
                    diff = (slice_size - z2 + z1) // 2
                    z1 = max(0, z1 - diff)
                    z2 = z1 + slice_size
                images = image_cube[z1:z2, y1 * 2:y2 * 2, x1 * 2:x2 * 2].copy()
                masks = mask_cube[z1:z2, y1:y2, x1:x2].copy()
                slice_size = slice_size

                replay = None
                image_crops = []
                mask_crops = []
                for i in range(images.shape[0]):
                    image = images[i]
                    mask = masks[i]
                    h, w, = mask.shape
                    mask = cv2.resize(mask, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
                    if replay is None:
                        sample = crop_augs(image=image, mask=mask)
                        replay = sample["replay"]
                    else:
                        sample = ReplayCompose.replay(replay, image=image, mask=mask)
                    image_ = sample["image"]
                    image_crops.append(image_)
                    mask_crops.append(sample["mask"])
                images = np.array(image_crops).astype(np.float32)
                masks = np.array(mask_crops).astype(np.float32)
                images = np.expand_dims(images, -1)
                masks = np.expand_dims(masks, -1)
                images = (images - image_mean) / image_std

                images = np.concatenate([images, images, masks], axis=-1)
                h = images.shape[0]
                if h > slice_size:
                    images = images[: slice_size]
                    all_images.append(np.moveaxis(images, -1, 0))
                    images = images[-slice_size:]
                    all_images.append(np.moveaxis(images, -1, 0))
                else:
                    if h != slice_size:
                        tmp = np.zeros((slice_size, *images.shape[1:]))
                        tmp[:h] = images
                        images = tmp
                    all_images.append(np.moveaxis(images, -1, 0))

        sample = {}
        sample['image'] = torch.from_numpy(np.array(all_images)).float()
        return sample

    def _predict_seg_mask(self, data) -> np.ndarray:
        sample = self._get_image_cube_segmentation(data)

        image = sample["image"]
        h = int(sample["h"])
        imgs = image.cpu().float().unsqueeze(0)

        case_preds = np.zeros((imgs.shape[2], 256, 256), dtype=np.float32)

        with torch.no_grad():
            slices = get_slices(imgs, dim=2, window=256, overlap=128)
            for slice in slices:
                batch = imgs[:, :, slice.i_from:slice.i_to].to(self.device).float()
                preds = torch.softmax(self.seg_model(batch)["mask"], dim=1)[0]
                preds = torch.argmax(preds, dim=0)
                preds = preds.cpu().numpy()

                for pred_idx in range(slice.i_start, preds.shape[0]):
                    idx = slice.i_from + pred_idx
                    y_pred = preds[pred_idx]
                    case_preds[idx] = y_pred[:, :]
        case_preds = np.array(case_preds)[:h]
        case_preds = case_preds.astype(np.uint8)
        return case_preds

    def _predict_cls(self, data, mask_cube):
        sample = self._get_image_cubes_classification(data, mask_cube)
        image = sample["image"]
        imgs = image.cpu().float()

        def predict_model(model):
            preds = []
            with torch.no_grad():
                for i in range(len(imgs)):
                    output = model(imgs[i:i + 1].to(self.device))#["cls"][0]
                    pred_slice = torch.sigmoid(output.float()).cpu().numpy().astype(np.float32)
                    output = model(torch.flip(imgs[i:i + 1].to(self.device), dims=(-1,)))#["cls"][0]
                    pred_slice += torch.sigmoid(output.float()).cpu().numpy().astype(np.float32)
                    pred_slice /= 2
                    preds.append(pred_slice)
            preds = np.max(np.array(preds), axis=0)
            preds[np.isnan(preds)] = 0.01
            return preds

        with torch.no_grad():
            preds = []
            for model in self.cls_models:
                preds.append(predict_model(model))
            preds = np.average(np.array(preds), axis=0)
            preds = np.clip(preds, 0.01, 0.99)
        return preds

    def predict(self, data):
        mask_cube = self._predict_seg_mask(data)
        prediction = self._predict_cls(data, mask_cube)
        # Todo: add all required information
        return {
            "prediction": prediction,
        }


if __name__ == '__main__':
    model = MDAIModel()
    # preds = model.predict({"scan_dir": "some_test_dir"})
    # print(preds)
    pass