import glob

import numpy as np
import pydicom
from skimage import img_as_ubyte
from torch.utils.data import Dataset
from tqdm import tqdm


def read_dicom(path, fix_monochrome=True):
    dicom = pydicom.read_file(path)

    return dicom, 0

    data = dicom.pixel_array

    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    meta = dicom.copy()

    del meta.PixelData

    return data, meta


def preprocess(img):
    a = img.copy()

    if np.min(img) > 0:
        img -= np.min(img)
    if np.min(img) < 0:
        img = img + abs(np.min(img))

    img = img / np.max(img)

    # RE: BECAUSE I DON'T KNOW WTF WAS WRONG
    if np.min(img) < 0:
        img += abs(np.min(img))
        img = img / np.max(img)

    b = img.copy()

    try:
        return img_as_ubyte(img)
    except:
        print(img.shape)
        print(img.dtype)
        print(np.min(img), np.max(img))

        print(a.dtype)
        print(np.min(a), np.max(a))

        print(b.dtype)
        print(np.min(b), np.max(b))


def read_dicom_image(path):
    img = pydicom.dcmread(path).pixel_array.astype(np.float32)
    return preprocess(img)


def load_dicom_array(f):
    dicom_files = glob.glob(f"{f}/*.dcm")
    dicoms = [pydicom.dcmread(d) for d in tqdm(dicom_files, total=len(dicom_files))]
    M = float(dicoms[0].RescaleSlope)
    B = float(dicoms[0].RescaleIntercept)
    # Assume all images are axial
    z_pos = np.array([float(d.ImagePositionPatient[-1]) for d in dicoms])  # different from patients
    z_inter = np.sort(z_pos)[3] - np.sort(z_pos)[2]

    dicoms = np.asarray([d.pixel_array for d in dicoms])
    dicoms = dicoms[np.argsort(-z_pos)]
    dicoms = dicoms * M
    dicoms = dicoms + B
    return dicoms, np.asarray(dicom_files)[np.argsort(-z_pos)], z_inter


def read_dicom_images(paths):
    dicoms = [pydicom.dcmread(path) for path in paths]
    M = float(dicoms[0].RescaleSlope)
    B = float(dicoms[0].RescaleIntercept)

    z_pos = np.array([float(dicom.ImagePositionPatient[-1]) for dicom in dicoms])
    z_inter = np.sort(z_pos)[3] - np.sort(z_pos)[2]

    dicoms = np.asarray([d.pixel_array for d in dicoms])
    dicoms = dicoms[np.argsort(-z_pos)]
    dicoms = dicoms * M
    dicoms = dicoms + B
    dicoms = np.stack([preprocess(d) for d in dicoms])
    return dicoms, np.asarray(paths)[np.argsort(-z_pos)]


def read_meta(path):
    meta = pydicom.dcmread(path, stop_before_pixels=True)
    return [float(meta.SliceThickness)] + [float(x) for x in list(meta.ImagePositionPatient)]


class LoadDicoms(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]

        dicom = pydicom.dcmread(file)

        z_pos = float(dicom.ImagePositionPatient[-1])

        dicom = preprocess(dicom.pixel_array)

        return dicom, z_pos