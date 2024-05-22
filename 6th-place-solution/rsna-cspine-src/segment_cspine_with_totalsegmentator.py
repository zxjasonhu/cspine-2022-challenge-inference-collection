import dicom2nifti
import glob
import nibabel
import numpy as np
import os
import os.path as osp

from scipy.ndimage.interpolation import zoom
from tqdm import tqdm


def create_dir(d):
    if not osp.exists(d):
        os.makedirs(d)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TMP_SAVE_SEG_DIR = "segmentations-two"
SAVE_FINAL_SEG_DIR = "../data/totalsegmentator-cspine-segmentations/"

create_dir(SAVE_FINAL_SEG_DIR)

cspine = glob.glob("../data/train_images/*")
#cspine = cspine[(len(cspine) // 2):]
for each_dicom_folder in tqdm(cspine):
    filepath = f"{SAVE_FINAL_SEG_DIR}/{each_dicom_folder.split('/')[-1]}.npy"
    if osp.exists(filepath):
        continue
    nii_file = dicom2nifti.convert_directory(each_dicom_folder, ".", compression=False, reorient=True)
    _ = os.system(f"TotalSegmentator -i {nii_file} -o {TMP_SAVE_SEG_DIR}")
    _ = os.system(f"rm {nii_file}")
    for label_index, level in enumerate([f"C{i+1}" for i in range(7)] + [f"T{i+1}" for i in range(12)]):
        pred_seg = nibabel.load(f"{TMP_SAVE_SEG_DIR}/vertebrae_{level}.nii").get_fdata()[:, :, ::-1]
        if level == "C1":
            spine_seg = np.zeros_like(pred_seg)
        spine_seg[pred_seg == 1] = label_index + 1
    scale_factor = [192. / spine_seg.shape[0], 192. / spine_seg.shape[1], 192. / spine_seg.shape[2]]
    spine_seg = zoom(spine_seg, scale_factor, order=0, prefilter=False)
    spine_seg = np.rot90(spine_seg.transpose(1, 2, 0), axes=(1, 0))
    spine_seg[spine_seg > 8] = 8
    np.save(filepath, spine_seg.astype("uint8"))

import numpy as np
import glob
x = glob.glob("totalsegmentator-cspine-segmentations/*")
from tqdm import tqdm
for i in tqdm(x):
    y = np.load(i)
    y[y>8] = 8
    np.save(i, y)