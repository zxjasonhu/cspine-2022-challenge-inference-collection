import numpy as np

from monai import transforms
from numpy import pi


class SomeOf:

    def __init__(self, augmentations, n, p, replace=False):
        self.augmentations = augmentations
        self.n = n
        self.p = p
        self.replace = replace

    def __call__(self, data):
        sampled_augs = np.random.choice(self.augmentations, self.n, replace=self.replace)
        for each_aug in sampled_augs:
            data = each_aug(data)
        return data


def resize_3d(imsize, size_mode="all", interp_mode=["trilinear", "nearest"]):
    return transforms.Resized(keys=["image", "label"], spatial_size=imsize, size_mode=size_mode,
                              mode=interp_mode, allow_missing_keys=True)


def crop_3d(imsize, mode="train"):
    if mode == "train":
        return transforms.RandSpatialCropd(keys=["image", "label"], roi_size=imsize, random_size=False, allow_missing_keys=True)
    else:
        return transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=imsize, allow_missing_keys=True)


def rand_augment_3d(n, p, replace=False):
    augmentations = [
        transforms.RandRotated(keys=["image", "label"], range_x=pi / 6., range_y=pi / 6., range_z=pi / 6., prob=1, allow_missing_keys=True),
        transforms.RandAffined(keys=["image", "label"], scale_range=(-0.2, 0.2), prob=1, allow_missing_keys=True),
        transforms.RandGridDistortiond(keys=["image", "label"], prob=1, allow_missing_keys=True),
        transforms.RandGaussianNoised(keys=["image", "label"], prob=1, allow_missing_keys=True),
        transforms.RandGaussianSmoothd(keys=["image", "label"], prob=1, allow_missing_keys=True),
        transforms.RandAdjustContrastd(keys=["image", "label"], prob=1, allow_missing_keys=True)
    ]
    return SomeOf(augmentations, n, p, replace)

