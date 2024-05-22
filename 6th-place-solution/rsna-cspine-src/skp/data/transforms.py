import albumentations as A
import cv2
import numpy as np
import torch

from .rand_augment import RandAugment


ADDITIONAL_TARGETS = {f'image{_}' : 'image' for _ in range(1,600)}
ADDITIONAL_TARGETS.update({f'mask{_}' : 'mask' for _ in range(1,600)})


# Preserves aspect ratio
# Pads the rest
def resize(imsize):
    x, y = imsize
    return A.Compose([
            A.LongestMaxSize(max_size=max(x,y), always_apply=True, p=1),
            A.PadIfNeeded(min_height=x, min_width=y, always_apply=True, p=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
        ], p=1)


# This can be more useful if training w/ crops and rectangular images
def resize_alt(imsize):
    x, y = imsize
    return A.Compose([
            A.SmallestMaxSize(max_size=max(x,y), always_apply=True, p=1)
        ], p=1)


# Ignore aspect ratio
def resize_ignore(imsize):
    x, y = imsize
    return A.Compose([
            A.Resize(imsize[0], imsize[1])
        ], p=1)


def crop(imsize, mode):
    x, y = imsize
    if mode == 'train':
        cropper = A.RandomCrop(height=x, width=y, always_apply=True, p=1)
    else:
        cropper = A.CenterCrop(height=x, width=y, always_apply=True, p=1)
    return A.Compose([
            cropper
        ], p=1, additional_targets=ADDITIONAL_TARGETS)


# Akin to RandAugment strategy
# p defines the probability that an augmentation will be applied
# n defines the number of augmentations to be applied
def simple_augment(p, n):
    augs = A.OneOf([
        A.RandomGamma(),
        A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0.0),
        A.RandomBrightnessContrast(contrast_limit=0.0, brightness_limit=0.2),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.0, rotate_limit=0,  border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0,  border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT),
        A.GaussianBlur(),
        A.GaussNoise()

    ], p=1)
    return A.Compose([augs] * n, p=p, additional_targets=ADDITIONAL_TARGETS)


def spatial_augment(p, n):
    augs = A.OneOf([
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.0, rotate_limit=0,  border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0,  border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT),
        A.GaussianBlur(),
        A.GaussNoise()
    ], p=1)
    return A.Compose([augs] * n, p=p, additional_targets=ADDITIONAL_TARGETS)


def simple_augment_with_dropout(p, n):
    augs = A.OneOf([
        A.RandomGamma(),
        A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0.0),
        A.RandomBrightnessContrast(contrast_limit=0.0, brightness_limit=0.2),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.0, rotate_limit=0,  border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0,  border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT),
        A.GaussianBlur(),
        A.GaussNoise(),
        A.CoarseDropout(max_height=0.2, max_width=0.2, min_height=0.02, min_width=0.02, fill_value=0)
    ], p=1)
    return A.Compose([augs] * n, p=p, additional_targets=ADDITIONAL_TARGETS)


def spatial_augment_with_dropout(p, n):
    augs = A.OneOf([
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.0, rotate_limit=0,  border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0,  border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT),
        A.GaussianBlur(),
        A.GaussNoise(),
        A.CoarseDropout(max_height=0.2, max_width=0.2, min_height=0.02, min_width=0.02, fill_value=0)
    ], p=1)
    return A.Compose([augs] * n, p=p, additional_targets=ADDITIONAL_TARGETS)


class Preprocessor(object):
    """
    Object to deal with preprocessing.
    Easier than defining a function.
    """
    def __init__(self, image_range, input_range, mean, sdev, channels_last=True, reverse_channels=True):
        self.image_range = image_range
        self.input_range = input_range
        self.mean = mean 
        self.sdev = sdev
        self.channels_last = channels_last
        self.reverse_channels = reverse_channels

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = img.astype("float")
        elif isinstance(img, torch.Tensor):
            img = img.float()

        # Preprocess an input image
        image_min = float(self.image_range[0])
        image_max = float(self.image_range[1])
        model_min = float(self.input_range[0])
        model_max = float(self.input_range[1])
        image_range = image_max - image_min
        model_range = model_max - model_min 
        img = (((img - image_min) * model_range) / image_range) + model_min 

        if self.channels_last:
            # Channels LAST format
            if img.shape[-1] == 3:
                if self.reverse_channels:
                    img = np.ascontiguousarray(img[..., ::-1])
                img[..., 0] -= self.mean[0]
                img[..., 1] -= self.mean[1]
                img[..., 2] -= self.mean[2]
                img[..., 0] /= self.sdev[0]
                img[..., 1] /= self.sdev[1]
                img[..., 2] /= self.sdev[2]
            else:
                avg_mean = np.mean(self.mean)
                avg_sdev = np.mean(self.sdev)
                img -= avg_mean
                img /= avg_sdev

        else:
            # Channels FIRST format
            if img.shape[1] == 3:
                if self.reverse_channels:
                    img = img[:,[2,1,0]]
                img[:, 0] -= self.mean[0]
                img[:, 1] -= self.mean[1]
                img[:, 2] -= self.mean[2]
                img[:, 0] /= self.sdev[0]
                img[:, 1] /= self.sdev[1]
                img[:, 2] /= self.sdev[2]
            else:
                avg_mean = np.mean(self.mean)
                avg_sdev = np.mean(self.sdev)
                img -= avg_mean
                img /= avg_sdev 

        return img

    def denormalize(self, img):
        assert self.channels_last
        # img.shape = (H, W, 3)
        img = img[..., ::-1]
        img[..., 0] *= self.sdev[0]
        img[..., 1] *= self.sdev[1]
        img[..., 2] *= self.sdev[2]
        img[..., 0] += self.mean[0]
        img[..., 1] += self.mean[1]
        img[..., 2] += self.mean[2]

        image_min = float(self.image_range[0])
        image_max = float(self.image_range[1])
        model_min = float(self.input_range[0])
        model_max = float(self.input_range[1])
        image_range = image_max - image_min
        model_range = model_max - model_min 

        img = ((img - model_min) * image_range) / model_range + image_min 
        return img

