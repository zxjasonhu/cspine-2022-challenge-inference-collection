import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd

from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from utils import create_dir


IMAGE_SAVEDIR = "../../data/train-numpy-seg-blocks-128/"
LABEL_SAVEDIR = "../../data/label-numpy-seg-blocks-128/"
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
    for ind, i in enumerate(range(0, 256-128, 32)):
        img_block = img_array[:, :, i:i+128].transpose(2, 0, 1)
        seg_block = seg_array[:, :, i:i+128].transpose(2, 0, 1)
        assert img_block.shape == seg_block.shape == (128, 256, 256)
        np.save(osp.join(IMAGE_SAVEDIR, f"{study_uid}-BLOCK-{ind:03d}.npy"), img_block.astype("uint8"))
        np.save(osp.join(LABEL_SAVEDIR, f"{study_uid}-BLOCK-{ind:03d}.npy"), seg_block.astype("uint8"))
