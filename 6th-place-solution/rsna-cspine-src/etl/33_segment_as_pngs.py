import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
sys.path.insert(0, "../../src/")
from skp import builder
from tqdm import tqdm
from utils import create_dir


def rescale(x):
    x = x / x.max()
    x = x - 0.5
    x = x * 2.0
    return x


def add_buffer(x1, x2, max_dist, buff=0.1):
    add_dist = int(buff * (x2 - x1))
    x1, x2 = x1 - add_dist, x2 + add_dist
    x1 = max(0, x1)
    x2 = min(max_dist, x2)
    return x1, x2


def plot_volume(img, num_row, num_col, sagittal=True):
    index = 1
    img_shape = img.shape[2] if sagittal else img.shape[0]
    for i in np.linspace(0, img_shape - 1, num_row * num_col).astype("int"):
        plt.subplot(num_row, num_col, index)
        plt.imshow(img[:, :, i] if sagittal else img[i], cmap="gray")
        index += 1
    plt.show()


def segment_two_stage(X, input_size1, input_size2, model_list1, model_list2, plot=False):
    # Assumes X is rescaled, a torch tensor, and has dimensions (Z, H, W)
    # X should NOT be resized
    assert isinstance(model_list1, list) and isinstance(model_list2, list)
    Z, H, W = X.size()
    rescale_factors = [input_size1[0] / Z, input_size1[1] / H, input_size1[2] / W]

    # Run first model
    pseg = torch.cat([model(F.interpolate(X.unsqueeze(0).unsqueeze(0), size=input_size1, mode="nearest"))
                      for model in model_list1]).mean(0)
    pseg = torch.sigmoid(pseg)
    # pseg.shape = (8, Z, H, W)

    # Get cervical spine boundaries and single-channel cervical spine level map
    p_cervical = pseg[:7].sum(0)
    spine_map = torch.argmax(pseg[:7], dim=0) + 1
    del pseg
    spine_map[p_cervical < 0.5] = 0

    # Get coordinates to crop input for second model
    coords = np.vstack(np.where(p_cervical.cpu().numpy() >= 0.25))
    del p_cervical
    # Get coordinates before rescaling for the spine map
    # That way, we don't have to resize the spine map back to original and resize it again to
    # model input size
    z1, z2, h1, h2, w1, w2 = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max(),\
                             coords[2].min(), coords[2].max()
    z1, z2 = add_buffer(z1, z2, input_size1[0], 0.1)
    h1, h2 = add_buffer(h1, h2, input_size1[1], 0.1)
    w1, w2 = add_buffer(w1, w2, input_size1[2], 0.1)

    spine_map = (spine_map[z1:z2+1, h1:h2+1, w1:w2+1].unsqueeze(0).unsqueeze(0) * 255 / 7).long()

    if plot:
        plotseg = spine_map.squeeze(0).squeeze(0).cpu().numpy()
        plot_volume(plotseg, 6, 4, sagittal=True)

    # Then get coordinates after rescaling, to crop the original input
    coords = coords.astype("float")
    coords[0] /= rescale_factors[0]
    coords[1] /= rescale_factors[1]
    coords[2] /= rescale_factors[2]
    coords = coords.astype("int")
    z1, z2, h1, h2, w1, w2 = coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max(),\
                             coords[2].min(), coords[2].max()
    z1, z2 = add_buffer(z1, z2, Z, 0.1)
    h1, h2 = add_buffer(h1, h2, H, 0.1)
    w1, w2 = add_buffer(w1, w2, W, 0.1)
    crop_cervical_coords = [z1, z2, h1, h2, w1, w2]

    X = X[z1:z2+1, h1:h2+1, w1:w2+1].unsqueeze(0).unsqueeze(0)
    Z, H, W = X.shape[2:]

    if plot:
        plotseg = X.squeeze(0).squeeze(0).cpu().numpy()
        plot_volume(plotseg, 6, 4, sagittal=True)

    X = torch.cat([F.interpolate(X, size=input_size2, mode="nearest"),
                   F.interpolate(rescale(spine_map.float()), size=input_size2, mode="nearest")], dim=1)
    # X.shape = (1, 2, Z, H, W)

    rescale_factors = [input_size2[0] / Z, input_size2[1] / H, input_size2[2] / W]

    # Run second model
    pseg = torch.cat([model(X) for model in model_list2]).mean(0)
    pseg = torch.sigmoid(pseg).cpu().numpy()

    if plot:
        plotseg = np.argmax(pseg, axis=0) + 1
        plotseg[pseg.sum(0) < 0.5] = 0
        plot_volume(plotseg, 6, 4, sagittal=True)

    coords_dict = {}
    threshold = 0.5
    for level in range(pseg.shape[0]):
        coords = np.vstack(np.where(pseg[level] >= threshold))
        if coords.shape[1] == 0:
            coords_dict[level] = None
            continue
        coords = coords.astype("float")
        coords[0] /= rescale_factors[0]
        coords[1] /= rescale_factors[1]
        coords[2] /= rescale_factors[2]
        coords = coords.astype("int")
        z1, z2, h1, h2, w1, w2 = coords[0].min(), coords[0].max(),\
                                 coords[1].min(), coords[1].max(),\
                                 coords[2].min(), coords[2].max()
        coords_dict[level] = (z1, z2, h1, h2, w1, w2)

    # Get cervical spine boundaries and single-channel cervical spine level map
    pseg = torch.from_numpy(pseg)
    p_cervical = pseg[:7].sum(0)
    spine_map = torch.argmax(pseg[:7], dim=0) + 1
    del pseg
    spine_map[p_cervical < 0.5] = 0

    return coords_dict,\
           crop_cervical_coords,\
           (F.interpolate(spine_map.unsqueeze(0).unsqueeze(0).float(), size=(Z, H, W), mode="nearest").squeeze(0).squeeze(0) * 255 / 7).cpu().numpy().astype("uint8")


torch.set_grad_enabled(False)

CONFIG1 = "../configs/seg/seg100.yaml"
CONFIG2 = "../configs/seg/seg101.yaml"

PNG_DIR = "../../data/pngs"

SAVE_DIR1 = "../../data/train-individual-vertebrae-cropped-pngs"
SAVE_DIR2 = "../../data/train-individual-vertebrae-masked"

SAVE_COORDS_FILE = "../../data/train_cspine_cascade_coords.pkl"

create_dir(SAVE_DIR1)
create_dir(SAVE_DIR2)

config1 = OmegaConf.load(CONFIG1)
config2 = OmegaConf.load(CONFIG2)

checkpoint_paths1 = np.sort(glob.glob("../../experiments/seg100/sbn/fold*/checkpoints/best.ckpt"))[:4]
checkpoint_paths2 = np.sort(glob.glob("../../experiments/seg101/sbn/fold*/checkpoints/best.ckpt"))[:4]

model_list1 = []
for fold, path in enumerate(checkpoint_paths1):
    tmp_config = config1.copy()
    tmp_config.model.load_pretrained = str(path)
    model_list1.append(builder.build_model(tmp_config).eval().cuda())

model_list2 = []
for fold, path in enumerate(checkpoint_paths2):
    tmp_config = config2.copy()
    tmp_config.model.load_pretrained = str(path)
    model_list2.append(builder.build_model(tmp_config).eval().cuda())


df = pd.read_csv("../../data/train_metadata.csv")
studies = list(np.sort(df.StudyInstanceUID.unique()))
df = df[df.StudyInstanceUID.isin(studies)]

vertebra_level_dict = {}
for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    study_files = study_df.filename
    study_files = [_.replace("dcm", "png") for _ in study_files]
    study_stack = np.stack([cv2.imread(osp.join(PNG_DIR, fi), cv2.IMREAD_UNCHANGED) for fi in study_files])
    coords_dict, crop_cervical_coords, spine_map = segment_two_stage(rescale(torch.from_numpy(study_stack).float()).cuda(),
                                                                     (192, 192, 192), (192, 192, 192),
                                                                     model_list1, model_list2, plot=False)

    z1, z2, h1, h2, w1, w2 = crop_cervical_coords
    cropped_stack = study_stack[z1:z2+1, h1:h2+1, w1:w2+1]
    fused = np.stack([cropped_stack, cropped_stack, spine_map], axis=-1)
    plot_volume(fused, 6, 4)
    cropped_files = study_files[z1:z2+1]
    create_dir(osp.join(SAVE_DIR1, study_id))
    vertebra_level_dict[study_id] = {}
    vertebra_level_dict[study_id]["filenames"] = cropped_files
    vertebra_level_dict[study_id]["level"] = [np.unique(_) for _ in spine_map]
    for cropped_fi, each_image in zip(cropped_files, fused):
        status = cv2.imwrite(osp.join(SAVE_DIR1, study_id, cropped_fi.split("/")[-1]), each_image)


with open("../../data/train_vertebra_level_dict.pkl", "wb") as f:
    pickle.dump(vertebra_level_dict, f)




