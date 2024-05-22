import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd

from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from utils import create_dir


IMAGE_SAVEDIR = "../../data/train-numpy-seg-pngs/"
LABEL_SAVEDIR = "../../data/label-numpy-seg-pngs/"
create_dir(IMAGE_SAVEDIR)
create_dir(LABEL_SAVEDIR)

segmentations = glob.glob("../../data/segmentations-numpy/*.npy")
df = pd.read_csv("../../data/train_metadata.csv")

for seg in tqdm(segmentations, total=len(segmentations)):
    study_uid = seg.split("/")[-1].replace(".npy", "")
    tmp_df = df[df.StudyInstanceUID == study_uid].sort_values("ImagePositionPatient_2", ascending=False)
    study_images = tmp_df.filename.tolist()
    study_images = [osp.join("../../data/pngs", i.replace("dcm", "png")) for i in study_images]
    seg_array = np.load(seg)
    assert len(seg_array) == len(study_images)
    img_array = np.stack([cv2.imread(i, 0) for i in study_images])
    assert seg_array.shape == img_array.shape
    z, x, y = img_array.shape
    rescale_factor = [256. / z, 256. / x, 256. / y]
    seg_array = zoom(seg_array, rescale_factor, order=0, prefilter=False)
    img_array = zoom(img_array, rescale_factor, order=0, prefilter=False)
    assert seg_array.shape == img_array.shape == (256, 256, 256)
    # Dimensions are (Z, H, W) - slice through LAST dimension to get sagittal slices
    for i in range(seg_array.shape[-1]):
        imgfile = osp.join(IMAGE_SAVEDIR, f"{study_uid}-SLICE-{i:03d}.png")
        segfile = imgfile.replace(IMAGE_SAVEDIR, LABEL_SAVEDIR)
        _ = cv2.imwrite(imgfile, img_array[:,:,i].astype("uint8"))
        _ = cv2.imwrite(segfile, seg_array[:,:,i].astype("uint8"))

