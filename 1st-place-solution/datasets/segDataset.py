import numpy as np
import pydicom
import cv2
import torch
from torch.utils.data import Dataset
from glob import glob
import os

image_size_seg = (128, 128, 128)

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (image_size_seg[0], image_size_seg[1]), interpolation=cv2.INTER_AREA)
    return data


def load_dicom_line_par(path):    
    t_paths = sorted(glob(os.path.join(path, "*")), key=lambda x: int(x.split('/')[-1].split(".")[0].split("-")[-1]))

    n_scans = len(t_paths)
    #     print(n_scans)
    indices = np.quantile(list(range(n_scans)), np.linspace(0., 1., image_size_seg[2])).round().astype(int)
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images.append(load_dicom(filename))
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images


class SegTestDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = load_dicom_line_par(row.image_folder)
        if image.ndim < 4:
            image = np.expand_dims(image, 0)
        image = image.astype(np.float32).repeat(3, 0)  # to 3ch
        image = image / 255.
        return torch.tensor(image).float()
