import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

from collections import defaultdict
from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from sklearn.metrics import roc_auc_score
from skp import builder
from tqdm import tqdm


torch.set_grad_enabled(False)


def window(x, WL, WW):
    upper, lower = WL+WW//2, WL-WW//2
    x = np.clip(x, lower, upper)
    x = x - lower
    x = x / (upper - lower)
    x = x * 255
    x = x.astype('uint8')
    return x


def load_dicom_volume(dicom_folder):
    dicom_files = glob.glob(osp.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(_) for _ in dicom_files]
    z_positions = [float(_.ImagePositionPatient[2]) for _ in dicoms]
    dicom_arrays = [_.pixel_array.astype("float32") for _ in dicoms]
    rescale_slope = float(dicoms[0].RescaleSlope)
    rescale_intercept = float(dicoms[0].RescaleIntercept)
    del dicoms

    # Deal with potential scenario where not all arrays are the same shape
    # This assumes that all arrays have the same number of dimensions (2)
    array_shapes = np.vstack([_.shape for _ in dicom_arrays])
    h, w = np.median(array_shapes[:,0]), np.median(array_shapes[:,1])
    for ind, arr in enumerate(dicom_arrays):
        if arr.shape[0] != h or arr.shape[1] != w:
            print("Mismatched shape, resizing ...")
            scale_h, scale_w = float(h) / arr.shape[0], float(w) / arr.shape[1]
            arr = zoom(arr, [scale_h, scale_w], order=1, prefilter=False)
            dicom_arrays[ind] = arr

    array = np.stack(dicom_arrays)
    del dicom_arrays
    array = rescale_slope * array + rescale_intercept
    array = window(array, WL=400, WW=2500)

    # Sort in DESCENDING order by z-position
    array = array[np.argsort(z_positions)[::-1]]
    return array


def plot_volume(array, skip=10, sagittal=False):
    length = array.shape[2] if sagittal else array.shape[0]
    for i in range(0, length, skip):
        image = array[..., i] if sagittal else array[i]
        if np.sum(image) == 0:
            continue
        plt.imshow(image, cmap="gray")
        plt.show()


def rescale(x):
    # Rescale to [-1, 1]
    x = x / x.max()
    x = x - 0.5
    x = x * 2
    return x


def unscale(x):
    x = x + 1
    x = x * 255 / 2
    return x


def load_models(config_file, checkpoint_folder, model_type="classification", cuda=True, load_indices=None):
    assert model_type in ["classification", "segmentation", "sequence"]
    config = OmegaConf.load(config_file)
    if model_type == "segmentation":
        config.model.params.encoder_params.pretrained = False
    elif model_type == "classification":
        config.model.params.pretrained = False
    checkpoints = np.sort(glob.glob(osp.join(checkpoint_folder, "*")))
    if isinstance(load_indices, (list, tuple)):
        load_indices = list(load_indices)
        checkpoints = checkpoints[load_indices]
    models = []
    for each_checkpoint in checkpoints:
        _config = config.copy()
        _config.model.load_pretrained = str(each_checkpoint)
        _model = builder.build_model(_config).eval()
        if cuda:
            _model = _model.cuda()
        models.append(_model)
    return models


def segment_cervical_spine(volume, inference_shape, segmentation_models, threshold, adjustment):
    orig_shape = volume.shape[2:]
    volume = F.interpolate(volume, size=inference_shape, mode="nearest")
    segmentation = torch.sigmoid(torch.cat([seg_model(volume.cuda()) for seg_model in segmentation_models])).mean(0)
    # Create a 1-channel cervical spine map
    p_spine = segmentation.sum(0)
    spine_map = torch.argmax(segmentation, dim=0) + 1
    spine_map[p_spine < threshold] = 0
    spine_map[spine_map == 8] = 0 # Get rid of thoracic spine
    spine_map = (spine_map * 255) / 7 # Rescale to an 8-bit image
    cspine_coords = {}
    print("Obtaining cervical spine coordinates ...")
    for level in range(7):
        coords = torch.stack(torch.where(segmentation[level] >= threshold)).cpu().numpy()
        coords[0] = coords[0] * orig_shape[0] / inference_shape[0]
        coords[1] = coords[1] * orig_shape[1] / inference_shape[1]
        coords[2] = coords[2] * orig_shape[2] / inference_shape[2]
        adjusted_threshold = threshold
        while coords.shape[1] == 0 and adjusted_threshold > adjustment:
            print(f"C{level+1} not found, lowering threshold to {adjusted_threshold - adjustment:0.1f} ...")
            adjusted_threshold -= adjustment
            adjusted_threshold = np.round(adjusted_threshold, 1)
            coords = torch.stack(torch.where(segmentation[level] >= threshold)).cpu().numpy()
        if coords.shape[1] == 0:
            print(f"Segmentation for C{level+1} failed !")
            cspine_coords[level] = None
        else:
            cspine_coords[level] = (coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max(), coords[2].min(), coords[2].max())
    return F.interpolate(spine_map.unsqueeze(0).unsqueeze(0), size=orig_shape, mode="nearest").squeeze(0).squeeze(0), cspine_coords


def center_crop(x, crop_size):
    h, w = crop_size
    orig_h, orig_w = x.shape[-2], x.shape[-1]
    diff_h, diff_w = (orig_h - h) // 2, (orig_w - w) // 2
    return x[..., diff_h:diff_h+h, diff_w:diff_w+w]


cspine_segmentation_models = load_models("configs/seg/pseudoseg000.yaml",
                                         "../kaggle-checkpoints/pseudoseg000.yaml",
                                         model_type="segmentation",
                                         load_indices=[0, 2, 4])


feature_extractors_3d = load_models("configs/chunk/chunk000.yaml",
                                    "../kaggle-checkpoints/chunk000/",
                                    load_indices=[0, 1, 2, 3, 4])

chunk_sequence_models = load_models("configs/chunkseq/chunkseq003.yaml",
                                    "../kaggle-checkpoints/chunkseq003/",
                                    model_type="sequence",
                                    load_indices=[0, 1, 2, 3, 4])


feature_extractors_2d = load_models("configs/cas/cas001.yaml",
                                    "../kaggle-checkpoints/cas001/",
                                    load_indices=[0, 1, 2, 3, 4])

slice_sequence_models = load_models("configs/casseq/casseq000.yaml",
                                    "../kaggle-checkpoints/casseq000/",
                                    model_type="sequence",
                                    load_indices=[0, 1, 2, 3, 4])


chunk_feature_extractors_2d = load_models("configs/cascrop/cascrop007.yaml",
                                          "../kaggle-checkpoints/cascrop007/",
                                          load_indices=[0, 1, 2, 3, 4])

chunk_feature_sequence_models = load_models("configs/caschunk/caschunk001.yaml",
                                            "../kaggle-checkpoints/caschunk001/",
                                            model_type="sequence",
                                            load_indices=[0, 1, 2, 3, 4])

chunk_slice_sequence_models = load_models("configs/chunkseq/chunkseq004.yaml",
                                          "../kaggle-checkpoints/chunkseq004/",
                                          model_type="sequence",
                                          load_indices=[0, 1, 2, 3, 4])


FOLD = 0
train_folds = pd.read_csv("../data/train_kfold.csv")
fold_df = train_folds[train_folds.outer == FOLD]
test_images = [osp.join("../data/train_images/", _) for _ in fold_df.StudyInstanceUID]
test_images = test_images[:10]
feature_extractors_3d = [feature_extractors_3d[FOLD]]
chunk_sequence_models = [chunk_sequence_models[FOLD]]

chunk_feature_extractors_2d = [chunk_feature_extractors_2d[FOLD]]
chunk_feature_sequence_models = [chunk_feature_sequence_models[FOLD]]
chunk_slice_sequence_models = [chunk_slice_sequence_models[FOLD]]

feature_extractors_2d = [feature_extractors_2d[FOLD]]
slice_sequence_models = [slice_sequence_models[FOLD]]

threshold = 0.4
adjustment = 0.1
segmentation_inference_size = (192, 192, 192)
chunk_inference_size = (64, 288, 288)
chunk_slice_inference_size = (32, 288, 288)
slice_inference_size = (640, 640)
slice_inference_crop_size = (560, 560)
slice_seq_len = 128

do_chunk_inference = True
do_slice_inference = True

chunk_prediction_dict, chunk_slice_prediction_dict, slice_prediction_dict = {}, {}, {}
for each_image in tqdm(test_images):
    print("Loading DICOM volume ...")
    X = load_dicom_volume(each_image)
    print("Rescaling to -1, 1 and converting from numpy to torch ...")
    X = rescale(X)
    X = torch.from_numpy(X).float().unsqueeze(0).unsqueeze(0)
    # X.shape = (1, 1, num_images, height, width)
    print("Segmenting cervical spine ...")
    spine_map, cspine_coords = segment_cervical_spine(X, segmentation_inference_size, cspine_segmentation_models, threshold, adjustment)
    if do_chunk_inference:
        chunk_features = defaultdict(list)
        chunk_slice_features = defaultdict(list)
        print("Extracting vertebra chunk features ...")
        for level, coords in cspine_coords.items():
            if not isinstance(coords, tuple):
                print(f"C{level+1} not found ... Using 0-vector ...")
                for fold, model in enumerate(feature_extractors_3d):
                    chunk_features[fold].append(torch.zeros((1, 432)).float().cuda())
                for fold, model in enumerate(chunk_feature_extractors_2d):
                    chunk_slice_features[fold].append(torch.zeros((1, 1280)).float().cuda())
            else:
                x1, x2, y1, y2, z1, z2 = coords
                orig_chunk = X[:, :, x1:x2, y1:y2, z1:z2]
                chunk = F.interpolate(orig_chunk, size=chunk_inference_size, mode="trilinear")
                for fold, model in enumerate(feature_extractors_3d):
                    chunk_features[fold].append(model.extract_features(chunk.cuda()))
                chunk_spine_map = spine_map[x1:x2, y1:y2, z1:z2].unsqueeze(0).unsqueeze(0)
                # chunk_spine_map.shape = (1, 1, z, h, w)
                chunk = torch.cat([orig_chunk.cuda(), orig_chunk.cuda(), rescale(chunk_spine_map)], dim=1)
                # chunk.shape = (1, 3, z, h, w)
                chunk = F.interpolate(chunk, size=chunk_slice_inference_size, mode="trilinear").squeeze(0).transpose(0, 1)
                # chunk.shape = (z, 3, h, w)
                for fold, model in enumerate(chunk_feature_extractors_2d):
                    next_input = (model.extract_features(chunk).unsqueeze(0), torch.ones((1, chunk_slice_inference_size[0])).cuda())
                    chunk_slice_features[fold].append(chunk_feature_sequence_models[fold].extract_features(next_input))
        for fold, features in chunk_features.items():
            chunk_features[fold] = (torch.cat(features).unsqueeze(0).cuda(), torch.ones((1, 7)).float().cuda())
        for fold, features in chunk_slice_features.items():
            chunk_slice_features[fold] = (torch.cat(features).unsqueeze(0).cuda(), torch.ones((1, 7)).float().cuda())
        print("Chunk sequence inference ...")
        chunk_pred_list = []
        for fold, model in enumerate(chunk_sequence_models):
            chunk_pred_list.append(torch.sigmoid(model(chunk_features[fold])).cpu().numpy())
        chunk_prediction_dict[each_image] = np.mean(np.stack(chunk_pred_list, axis=0), axis=0)
        chunk_slice_pred_list = []
        for fold, model in enumerate(chunk_slice_sequence_models):
            chunk_slice_pred_list.append(torch.sigmoid(model(chunk_slice_features[fold])).cpu().numpy())
        chunk_slice_prediction_dict[each_image] = np.mean(np.stack(chunk_slice_pred_list, axis=0), axis=0)
    if do_slice_inference:
        spine_present_on_slice = torch.where(spine_map.sum((1, 2)) > 0)[0]
        start_slice, end_slice = spine_present_on_slice.min().item(), spine_present_on_slice.max().item()
        spine_map = rescale(spine_map[start_slice:end_slice + 1]).unsqueeze(0).unsqueeze(0)
        X = X.cuda()
        X = torch.cat([X[:, :, start_slice:end_slice + 1], X[:, :, start_slice:end_slice + 1], spine_map], dim=1)
        del spine_map
        slice_indices = np.arange(X.size(2))
        slice_indices = zoom(slice_indices, [slice_seq_len / len(slice_indices)], order=0, prefilter=False)
        X = center_crop(F.interpolate(X[:, :, slice_indices],
                                      size=(slice_seq_len, slice_inference_size[0], slice_inference_size[1]),
                                      mode="trilinear"),
                        slice_inference_crop_size)
        X = X.squeeze(0).transpose(0, 1)
        slice_features = {}
        print("Extracting slice features ...")
        for fold, model in enumerate(feature_extractors_2d):
            tmp_features = [model.extract_features(X[i:i+16]) for i in range(0, len(X), 16)]
            tmp_features = torch.cat(tmp_features)
            slice_features[fold] = (tmp_features.unsqueeze(0).cuda(), torch.ones((1, len(tmp_features))).float().cuda())
        print("Slice sequence inference ...")
        slice_pred_list = []
        for fold, model in enumerate(slice_sequence_models):
            slice_pred_list.append(torch.sigmoid(model(slice_features[fold])).cpu().numpy())
        slice_prediction_dict[each_image] = np.mean(np.stack(slice_pred_list, axis=0), axis=0)
