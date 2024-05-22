import cv2
import glob
import random
import numpy as np
import os, os.path as osp
import re
import torch

from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from . import helper


class ImageDataset(Dataset):

    def __init__(self,
        inputs,
        labels,
        resize=None,
        augment=None,
        crop=None,
        preprocess=None,
        channels="bgr",
        flip=False,
        invert=False,
        reverse_channels=False,
        verbose=True,
        test_mode=False,
        return_name=False,
        return_imsize=False):

        self.inputs = inputs
        self.labels = labels 
        self.resize = resize 
        self.augment = augment 
        self.crop = crop 
        self.preprocess = preprocess
        assert channels.lower() in ["rgb", "bgr", "grayscale", "2dc"], f"{channels} is not a valid argument for `channels`"
        self.channels = channels.lower()
        self.flip = flip 
        self.invert = invert
        self.reverse_channels = reverse_channels
        self.verbose = verbose 
        self.test_mode = test_mode 
        self.return_name = return_name 
        self.return_imsize = return_imsize

    def __len__(self):
        return len(self.inputs)

    def process_image(self, X):
        if self.resize:
            X = self.resize(image=X)["image"]
        if self.augment:
            X = self.augment(image=X)["image"]
        if self.crop:
            X = self.crop(image=X)["image"]
        if self.invert:
            X = np.invert(X)
        if self.preprocess:
            X = self.preprocess(X)
        return X.transpose(2, 0, 1)

    @staticmethod
    def flip_array(X):
        # X.shape = (C, H, W)
        if random.random() > 0.5:
            X = X[:, :, ::-1]
        if random.random() > 0.5:
            X = X[:, ::-1, :]
        return np.ascontiguousarray(X)

    def load_image(self, i):
        if self.channels in ["bgr", "rgb"]: 
            X = cv2.imread(self.inputs[i])
            # OpenCV reads images as BGR
            if self.channels == "rgb": X = np.ascontiguousarray(X[:,:,::-1])
        elif self.channels == "grayscale":
            X = cv2.imread(self.inputs[i], 0)
            if isinstance(X, np.ndarray):
                X = np.expand_dims(X, axis=-1)
        elif self.channels == "2dc":
            # The input will be a string of filenames, separated by commas
            inputs = self.inputs[i].split(",")
            inputs = [cv2.imread(each_input, 0) if each_input != "placeholder" else None for each_input in inputs]
            for each_input in inputs:
                if isinstance(each_input, np.ndarray):
                    placeholder = np.zeros_like(each_input)
                    placeholder[...] -= np.min(each_input)
                    break
            for idx, each_input in enumerate(inputs):
                if not isinstance(each_input, np.ndarray):
                    inputs[idx] = placeholder
            X = np.concatenate([np.expand_dims(each_input, axis=-1) for each_input in inputs], axis=-1)
        if self.reverse_channels and not self.test_mode and X.shape[-1] > 1:
            if random.random() > 0.5:
                X = np.ascontiguousarray(X[..., ::-1])
        return X

    def get(self, i):
        try:
            X = self.load_image(i)

            if not isinstance(X, np.ndarray):
                print(f"OpenCV failed to load {self.inputs[i]} and returned `None`")
                return None 

            return X 

        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def __getitem__(self, i):
        X = self.get(i)
        while isinstance(X, type(None)):
            if self.verbose: print("Failed to read {} !".format(self.inputs[i]))
            i = np.random.randint(len(self))
            X = self.get(i)

        imsize = X.shape[:2]

        X = self.process_image(X)

        if self.flip and not self.test_mode:
            X = self.flip_array(X)

        y = self.labels[i]

        X = torch.tensor(X).float()
        y = torch.tensor(y).float() 

        out = [X, y]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)


class XR_DICOMDataset(ImageDataset):

    def __init__(self, *args, **kwargs):
        # Default is to convert to 8-bit image
        # However, can also leave in float if more granularity is desired
        # Values will be rescaled to [0, 1], then augmentations+preprocessing will be applied
        # Remember, many augmentations only work with 8-bit images
        self.convert_8bit = kwargs.pop("convert_8bit", True)
        super().__init__(*args, **kwargs)

    def load_dicom(self, dcmfile):
        return helper.load_dicom(dcmfile, mode="XR", convert_8bit=self.convert_8bit,
                                 verbose=self.verbose)

    def get(self, i):
        try:
            dicom = self.load_dicom(self.inputs[i])
            return dicom
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None 


class CT_DICOMDataset(XR_DICOMDataset):

    def __init__(self, *args, **kwargs):
        assert "window" in kwargs, f"You must specify a window or list of windows"
        # If specifying windows, it must be a LIST of LISTS
        # If only 1 window, should be in this format --> [[WL, WW]]
        # Note that `raw_hu` is an option if this is the desired input
        # However, many augmentations only work with uint8, so be aware of this
        self.window = kwargs.pop("window") 
        super().__init__(*args, **kwargs)

    def load_dicom(self, dcmfile):
        return helper.load_dicom(dcmfile, mode="CT", convert_8bit=self.convert_8bit,
                                 window=self.window, verbose=self.verbose)


class MR_DICOMDataset(XR_DICOMDataset):

    def load_dicom(self, dcmfile):
        return helper.load_dicom(dcmfile, mode="MR", convert_8bit=self.convert_8bit,
                                 verbose=self.verbose)


class ImageStackDataset(Dataset):

    def __init__(self,
        inputs,
        labels,
        num_images,
        pad_or_resample="resample",
        resize=None,
        augment=None,
        crop=None,
        preprocess=None,
        channels="bgr",
        flip=False,
        invert=False,
        verbose=True,
        test_mode=False,
        return_name=False,
        return_imsize=False):

        self.inputs = inputs
        self.labels = labels 
        self.num_images = num_images
        self.pad_or_resample = pad_or_resample
        self.resize = resize 
        self.augment = augment 
        self.crop = crop 
        self.preprocess = preprocess
        assert channels in ["rgb", "bgr", "grayscale"], f"{channels} is not a valid argument for `channels`"
        self.channels = channels
        self.flip = flip 
        self.invert = invert 
        self.verbose = verbose 
        self.test_mode = test_mode 
        self.return_name = return_name 
        self.return_imsize = return_imsize

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def flip_array(X):
        # X.shape = (C, Z, H, W)
        if random.random() > 0.5:
            X = X[:, :, :, ::-1]
        if random.random() > 0.5:
            X = X[:, :, ::-1, :]
        if random.random() > 0.5:
            X = X[:, ::-1, :, :]
        X = np.ascontiguousarray(X)
        return X

    def resample_slices(self, X):
        """ Not really used in favor of just changing which images to load,
        which is equivalent to nearest-neighbor resampling
        """
        new_size = (self.num_images, X.shape[-2], X.shape[-1])
        X = F.interpolate(X.unsqueeze(0), size=new_size, mode='nearest', align_corners=False)
        return X.squeeze(0)

    def pad_slices(self, X):
        if X.shape[1] > self.num_images:
            return self.resample_slices(X)
        filler = torch.zeros_like(X[1]).squeeze(0).repeat(1, self.num_images - X.shape[1], 1, 1)
        filler[...] = X.min()
        return torch.cat([X, filler], dim=1)

    def process_image(self, X):
        if self.resize: 
            X = np.asarray([self.resize(image=_)['image'] for _ in X])
        if X.ndim == 3: X = np.expand_dims(X, axis=-1)
        assert X.ndim == 4
        # X.shape (Z, H, W, C)
        if self.augment and not self.test_mode: 
            to_augment = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_augment.update({'image': X[0]})
            augmented = self.augment(**to_augment)
            X = np.asarray([augmented['image']] + [augmented['image{}'.format(_)] for _ in range(1,len(X))])
        if self.crop: 
            to_crop = {'image{}'.format(ind) : X[ind] for ind in range(1,len(X))}
            to_crop.update({'image': X[0]})
            cropped = self.crop(**to_crop)
            X = np.asarray([cropped['image']] + [cropped['image{}'.format(_)] for _ in range(1,len(X))])
        if self.invert: X = np.invert(X)
        if self.preprocess: X = self.preprocess(X)
        return X.transpose(3, 0, 1, 2)

    @staticmethod
    def check_if_image(fp):
        return "png" in fp or "jpg" in fp or "jpeg" in fp

    def load_image(self, fp):
        if self.channels in ["bgr", "rgb"]:
            X = cv2.imread(fp)
            if self.channels == "rgb": X = np.ascontiguousarray(X[:,:,::-1])
        elif self.channels == "grayscale":
            X = np.expand_dims(cv2.imread(fp, 0), axis=-1)            
        return X

    def get(self, i):
        try:
            slices = np.sort(glob.glob(osp.join(self.inputs[i], '*')))
            slices = [s for s in slices if self.check_if_image(s)]
            if len(slices) == 0:
                print(f"No images found for {self.inputs[i]}")
                return None
            if (self.pad_or_resample == "pad" and len(slices) > self.num_images) or \
                self.pad_or_resample == "resample":
                # If resampling, just resample the indices and load those images
                # Equivalent to nearest neighbors
                # If padding and the number of images exceeds the desired number,
                # resample the indices and load those images
                indices_to_load = zoom(np.arange(len(slices)), float(self.num_images) / len(slices), order=0, prefilter=False)
                slices = np.asarray([self.load_image(slices[ind_load]) for ind_load in indices_to_load])
            assert len(slices) == self.num_images
            return slices
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None 

    def __getitem__(self, i):
        X = self.get(i)
        while isinstance(X, type(None)):
            print(f"Failed to read {self.inputs[i]} !")
            i = np.random.randint(len(self))
            X = self.get(i) 

        X = self.process_image(X) 

        if self.flip and not self.test_mode:
            X = self.flip_array(X) 

        X = torch.tensor(X).float()
        if X.shape[1] != self.num_images and self.pad_or_sample == "pad":
            X = self.pad_slices(X) 

        y = torch.tensor(self.labels[i])
        return X, y 


class NumpyChunkDataset(ImageStackDataset):

    def pad_or_resample_array(self, array):
        if len(array) < self.num_images:
            if self.pad_or_resample == "pad":
                min_val = np.min(array)
                padding = np.zeros_like(array[0])
                padding = np.expand_dims(padding, axis=0)
                padding[...] = min(0, min_val)
                array = np.concatenate([array, padding])
            elif self.pad_or_resample == "resample":
                rescale = self.num_images / float(len(array))
                array = zoom(array, zoom=[rescale, 1, 1], order=0, prefilter=False)
        elif len(array) > self.num_images:
            if self.pad_or_resample == "pad":
                # Truncate
                array = array[:self.num_images]
            elif self.pad_or_resample == "resample":
                rescale = self.num_images / float(len(array))
                array = zoom(array, zoom=[rescale, 1, 1], order=0, prefilter=False)
        return array

    def get(self, i):
        try:
            array = np.load(self.inputs[i])
            array = self.pad_or_resample_array(array)
            assert len(array) == self.num_images
            return array
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None


class CT_DICOMStackDataset(ImageStackDataset):

    def __init__(self, *args, **kwargs):
        assert "window" in kwargs, f"You must specify a window or list of windows"
        # If specifying windows, it must be a LIST of LISTS
        # If only 1 window, should be in this format --> [[WL, WW]]
        # Note that `raw_hu` is an option if this is the desired input
        # However, many augmentations only work with uint8, so be aware of this
        self.convert_8bit = kwargs.pop("convert_8bit", True)
        self.window = kwargs.pop("window") 
        self.orientation = kwargs.pop("orientation", "axial")
        self.orientation_dict = {"coronal": 0, "sagittal": 1, "axial": 2}
        super().__init__(*args, **kwargs)

    def load_dicom(self, dcmfile):
        try:
            return helper.load_dicom(dcmfile, mode="CT", convert_8bit=self.convert_8bit,
                                     window=self.window, return_position=True, verbose=self.verbose)
        except Exception as e:
            if self.verbose: print(e)
            return None

    def get(self, i):
        try:
            # This assumes that the list of inputs given contains 
            # DICOM directories and that all files within the directory
            # are valid image DICOMs
            #
            # TODO: implement other variations, such as when the input
            # is a list of dicom files or a DataFrame containing 
            # the DICOM file paths
            dicoms = np.sort(glob.glob(osp.join(self.inputs[i], '*')))
            dicoms = [self.load_dicom(fp) for p in dicoms]
            dicoms = [d for d in dicoms if not isinstance(d, type(None))]
            if len(dicoms) == 0:
                print(f"No DICOMs found for {self.inputs[i]}")
                return None
            array, positions = [d[0] for d in dicoms], [d[1] for d in dicoms]
            positions = np.asarray(positions)
            # positions.shape = (num_images, 3)
            assert positions.shape[1] == 3 and positions.ndim == 2 
            positions = positions[:,self.orientation_dict[self.orientation]]
            positions = np.argsort(positions)
            array = np.concatenate(array, axis=0)
            array = array[positions]
            assert len(array) == self.num_images
            return array
        except Exception as e:
            if self.verbose: print(f'Failed to load {self.inputs[i]} :  {e}')
            return None 

    def __getitem__(self, i):
        X = self.get(i)
        while isinstance(X, type(None)):
            print(f"Failed to read {self.inputs[i]} !")
            i = np.random.randint(len(self))
            X = self.get(i) 

        X = self.process_image(X) 

        if self.flip and not self.test_mode:
            X = self.flip_array(X) 

        X = torch.tensor(X).float()
        if X.shape[1] != self.num_images:
            if self.pad_or_resample == "resample":
                X = self.resample_slices(X)
            elif self.pad_or_sample == "pad":
                X = self.pad_slices(X) 

        y = torch.tensor(self.labels[i])
        return X, y 


class MR_DICOMStackDataset(CT_DICOMStackDataset):

    def load_dicom(self, dcmfile):
        try:
            return helper.load_dicom(dcmfile, mode="MR", convert_8bit=self.convert_8bit,
                                     window=None, return_position=True, verbose=self.verbose)
        except Exception as e:
            if self.verbose: print(e)
            return None


class ImageSegmentDataset(ImageDataset):

    def __init__(self, *args, **kwargs):
        self.segmentation_format = kwargs.pop("segmentation_format", "png")
        self.add_foreground_channel = kwargs.pop("add_foreground_channel", False)
        self.one_hot_encode = kwargs.pop("one_hot_encode", False)
        # Only really used for PNG/JPG format
        self.num_classes = kwargs.pop("num_classes", None)
        if self.one_hot_encode: assert self.num_classes is not None
        self.max_255 = kwargs.pop("max_255", False)
        assert bool(re.search(r"png|jpg|jpeg|npy|numpy|rle|multislice_pred", self.segmentation_format))
        super().__init__(*args, **kwargs)

    def process_image(self, X, y):
        if self.resize: 
            X = self.resize(image=X)['image']
            y = self.resize(image=y)['image']
        if self.augment: 
            augmented = self.augment(image=X, mask=y)
            X, y = augmented['image'], augmented['mask']
        if self.crop: 
            cropped = self.crop(image=X, mask=y)
            X, y = cropped['image'], cropped['mask']
        if self.invert: X = np.invert(X)
        if self.preprocess: X = self.preprocess(X)
        X = X.transpose(2, 0, 1)
        if y.ndim == 3:
            y = y.transpose(2, 0, 1)
        return X, y

    @staticmethod
    def flip_array(X, y):
        # X.shape = (C, H, W)
        if random.random() > 0.5:
            X = X[:, :, ::-1]
            if y.ndim == 3:
                y = y[:, :, ::-1]
            elif y.ndim == 2:
                y = y[:, ::-1]
        if random.random() > 0.5:
            X = X[:, ::-1, :]
            if y.ndim == 3:
                y = y[:, ::-1, :]
            elif y.ndim == 2:
                y = y[::-1, :]
        if random.random() > 0.5 and X.shape[-1] == X.shape[-2]:
            X = X.transpose(0, 2, 1)
            if y.ndim == 3:
                y = y.transpose(0, 2, 1)
            elif y.ndim == 2:
                y = y.transpose(1, 0)
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        return X, y

    def load_segmentation(self, i):
        if self.labels[i] == 0: # this means we are doing inference and don't need segmentations
            return np.zeros((512,512,1))
        if self.segmentation_format in ["png","jpeg","jpg"]:
            assert self.num_classes is not None
            y = cv2.imread(self.labels[i], cv2.IMREAD_UNCHANGED)
            y = y[..., :self.num_classes].astype("float")
            if self.max_255: 
                y /= 255
        elif self.segmentation_format in ["npy", "numpy"]:
            y = np.load(self.labels[i])
        elif self.segmentation_format == "multislice_pred":
            fp = self.labels[i].split(",")
            if self.test_mode:
                y = cv2.imread(fp[1])
            else:
                y = [cv2.imread(_) for _ in fp]
                y = np.concatenate(y, axis=-1) 
        else:
            raise Exception(f"{self.segmentation_format} label format is not supported")
        
        if self.add_foreground_channel:
            if y.shape[-1] == 1:
                print("add_foreground_channel=True, however only 1 class present, so this will be ignored")
            else:
                fg = np.expand_dims(y.sum(-1), -1)
                fg[fg > 0] = 1

        return y

    def get(self, i):
        try:
            X = self.load_image(i)

            if not isinstance(X, np.ndarray):
                print(f"OpenCV failed to load {self.inputs[i]} and returned `None`")
                return None 

            y = self.load_segmentation(i)
            return X, y 

        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while isinstance(data, type(None)):
            if self.verbose: print("Failed to read {} !".format(self.inputs[i]))
            i = np.random.randint(len(self))
            data = self.get(i)

        X, y = data 

        imsize = X.shape[:2]

        X, y = self.process_image(X, y)
        if self.flip and not self.test_mode:
            X, y = self.flip_array(X, y)

        X = torch.tensor(X).float()
        y = torch.tensor(y).float() 
        if self.one_hot_encode:
            y = torch.nn.functional.one_hot(y.long(), num_classes=self.num_classes+1)[..., 1:].permute(2, 0, 1).float()

        out = [X, y]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)


class ImageStackSegmentDataset(ImageStackDataset):

    def __init__(self, *args, **kwargs):
        self.segmentation_format = kwargs.pop("segmentation_format", "png")
        self.add_foreground_channel = kwargs.pop("add_foreground_channel", False)
        self.crop_or_resample = kwargs.pop("crop_or_resample", "crop")
        self.one_hot_encode = kwargs.pop("one_hot_encode", False)
        # Only really used for PNG/JPG format
        self.num_classes = kwargs.pop("num_classes", None)
        self.max_255 = kwargs.pop("max_255", False)
        assert bool(re.search(r"png|jpg|jpeg|npy|numpy|rle", self.segmentation_format))
        if self.one_hot_encode: assert self.num_classes is not None
        super().__init__(*args, **kwargs)

    def process_image(self, X, y):
        if self.resize: 
            X = np.asarray([self.resize(image=_)['image'] for _ in X])
            y = np.asarray([self.resize(image=_)["image"] for _ in y])
        if X.ndim == 3: X = np.expand_dims(X, axis=-1)
        assert X.ndim == 4
        # X.shape (Z, H, W, C)
        if self.augment and not self.test_mode: 
            to_augment = {"image": X[0], "mask": y[0]}
            to_augment.update({f"image{ind}" : X[ind] for ind in range(1,len(X))})
            to_augment.update({f"mask{ind}"  : y[ind] for ind in range(1,len(y))})
            augmented = self.augment(**to_augment)
            X = np.asarray([augmented['image']] + [augmented['image{}'.format(_)] for _ in range(1,len(X))])
            y = np.asarray([augmented['mask']]  + [augmented['mask{}'.format(_)]  for _ in range(1,len(y))])
        if self.crop and not self.test_mode: 
            to_crop = {"image": X[0], "mask": y[0]}
            to_crop.update({f"image{ind}" : X[ind] for ind in range(1,len(X))})
            to_crop.update({f"mask{ind}"  : y[ind] for ind in range(1,len(y))})
            cropped = self.crop(**to_crop)
            X = np.asarray([cropped['image']] + [cropped['image{}'.format(_)] for _ in range(1,len(X))])
            y = np.asarray([cropped['mask']]  + [cropped['mask{}'.format(_)]  for _ in range(1,len(y))])
        if self.invert: X = np.invert(X)
        if self.preprocess: X = self.preprocess(X)
        X = X.transpose(3, 0, 1, 2)
        if y.ndim == 4:
            y = y.transpose(3, 0, 1, 2)
        return X, y

    @staticmethod
    def flip_array(X, y):
        # X.shape = (Z, H, W, C)
        if random.random() > 0.5:
            X = X[::-1, :, :]
            y = y[::-1, :, :]
        if random.random() > 0.5:
            X = X[:, ::-1, :]
            y = y[:, ::-1, :]
        if random.random() > 0.5:
            X = X[:, :, ::-1]
            y = y[:, :, ::-1]
        if random.random() > 0.5 and X.shape[-1] == X.shape[-2] == X.shape[-3]:
            # If volume is a cube, can rearrange the axes in any permutation
            axes = np.asarray([0, 1, 2])
            np.random.shuffle(axes)
            X = X.transpose(axes[0], axes[1], axes[2], 3)
            if y.ndim == 4:
                y = y.transpose(axes[0], axes[1], axes[2], 3)
            elif y.ndim == 3:
                y = y.transpose(axes[0], axes[1], axes[2])
        else:
            # If volume is not a cube, can only rearrange axes which are the same size
            # Most commonly, the axial dimensions will be the same size
            if random.random() > 0.5 and X.shape[-1] == X.shape[-2]:
                # Z,H,W -> Z,W,H
                X = X.transpose(0, 2, 1, 3)
                if y.ndim == 4:
                    y = y.transpose(0, 2, 1, 3)
                elif y.ndim == 3:
                    # y.shape = (Z, H, W)
                    y = y.transpose(0, 2, 1)
            if random.random() > 0.5 and X.shape[-1] == X.shape[-3]:
                # Z,H,W -> W,H,Z
                X = X.transpose(2, 1, 0, 3)
                if y.ndim == 4:
                    y = y.transpose(2, 1, 0, 3)
                elif y.ndim == 3:
                    y = y.transpose(2, 1, 0)
            if random.random() > 0.5 and X.shape[-2] == X.shape[-3]:
                # Z,H,W -> H,Z,W
                X = X.transpose(1, 0, 2, 3)
                if y.ndim == 4:
                    y = y.transpose(1, 0, 2, 3)
                elif y.ndim == 3:
                    y = y.transpose(1, 0, 2)
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        return X, y

    def load_segmentation(self, i):
        if self.labels[i] == 0: # this means we are doing inference and don't need segmentations
            return np.zeros((32,512,512,1))
        if self.segmentation_format in ["png","jpeg","jpg"]:
            assert self.num_classes is not None
            segfiles = np.sort(glob.glob(osp.join(self.labels[i], "*")))
            y = [np.expand_dims(cv2.imread(each_seg), axis=0) for each_seg in segfiles]
            y = np.concatenate(y)
            y = y[..., :self.num_classes].astype("float")
            if self.max_255: 
                y /= 255
        elif self.segmentation_format in ["npy", "numpy"]:
            y = np.load(self.labels[i])
        else:
            raise Exception(f"{self.segmentation_format} label format is not supported")

        return y

    def get(self, i):
        try:
            # Load segmentation
            y = self.load_segmentation(i)

            slices = np.sort(glob.glob(osp.join(self.inputs[i], '*')))
            slices = [s for s in slices if self.check_if_image(s)]
            if self.test_mode:
                # During inference, we load the whole volume without cropping
                X = np.asarray([self.load_image(s) for s in slices])
            else:
                if self.crop_or_resample == "resample":
                    # If resampling (not recommended), resample indices
                    # and load those images 
                    indices_to_load = zoom(np.arange(len(slices)), float(self.num_images) / len(slices), order=0, prefilter=False)
                    X = np.asarray([self.load_image(slices[ind_load]) for ind_load in indices_to_load])
                    y = y[indices_to_load]
                elif self.crop_or_resample == "crop":
                    if len(slices) <= self.num_images: 
                        # Just load all of them
                        X = np.asarray([self.load_image(s) for s in slices])
                    else:
                        # Randomly select start_index
                        start_index = np.random.randint(0, len(slices) - self.num_images + 1)
                        indices = np.arange(len(slices))
                        indices_to_load = indices[start_index: start_index+self.num_images]
                        X = np.asarray([self.load_image(slices[ind_load]) for ind_load in indices_to_load])
                        y = y[indices_to_load]

            if len(slices) == 0:
                print(f"No images found for {self.inputs[i]}")
                return None

            if not isinstance(X, np.ndarray):
                print(f"OpenCV failed to load {self.inputs[i]} and returned `None`")
                return None 

            # X and y should be in channels_last 
            if not self.test_mode:
                assert X.shape[:3] == y.shape[:3], f"image dimensions {X.shape[:3]} do not match label dimensions {y.shape[:3]}"

            return X, y 

        except Exception as e:
            if self.verbose:
               print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while isinstance(data, type(None)):
            if self.verbose: print("Failed to read {} !".format(self.inputs[i]))
            i = np.random.randint(len(self))
            data = self.get(i)

        X, y = data 
        imsize = X.shape[:2]

        if self.flip and not self.test_mode:
            X, y = self.flip_array(X, y)

        X, y = self.process_image(X, y)

        X = torch.tensor(X).float()
        y = torch.tensor(y).float() 
        if X.shape[1] < self.num_images and not self.test_mode:
            X = self.pad_slices(X)
            y = self.pad_slices(y)
        # Because of the way we load images, len(X) should never exceed num_images during training

        if self.one_hot_encode:
            # Exclude the background class by ignoring the first channel
            y = torch.nn.functional.one_hot(y.long(), num_classes=self.num_classes+1)[..., 1:].permute(3, 0, 1, 2).float()

        if self.add_foreground_channel:
            # y.shape = C, Z, H, W
            fg = y[0:7].sum(0).unsqueeze(0)
            y = torch.cat([y, fg])

        out = [X, y]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)


class NumpyChunkSegmentDataset(ImageStackSegmentDataset):
    """
    This dataset assumes that chunks have been premade and saved to disk 
    as Numpy arrays.
    """
    def load_segmentation(self, i):
        assert self.segmentation_format in ["npy", "numpy"]
        if self.labels[i] == 0: # this means we are doing inference and don't need segmentations
            return np.zeros((32,512,512,1))
        y = np.load(self.labels[i])

        return y

    def get(self, i):
        try:
            # Load segmentation
            y = self.load_segmentation(i)
            X = np.load(self.inputs[i])

            if X.ndim == 3:
                # No channel dimension
                X = np.expand_dims(X, axis=-1) 
                if self.channels != "grayscale":
                    # Assumes that you want fake RGB
                    X = np.repeat(X, 3, axis=-1)

            # X and y should be in channels_last 
            if not self.test_mode:
                assert X.shape[:3] == y.shape[:3], f"image dimensions {X.shape[:3]} do not match label dimensions {y.shape[:3]}"
            return X, y 

        except Exception as e:
            if self.verbose:
               print(e)
            return None

