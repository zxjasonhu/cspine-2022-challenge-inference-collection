from typing import List, Dict

import numpy as np

import cv2


def cam_to_intermediate_cam(cam: np.ndarray, interval: int, length: int = 40, h: int = 512,
                            w: int = 512, ) -> np.ndarray:
    grayscale_cam_resized = np.zeros((length, h, w))
    cam_length = cam.shape[0]
    frame_interval = cam_length / length
    max_index = cam_length - 1


    for i in range(length):
        # Find the indices of the original frames that we'll interpolate between
        idx1 = int(i / length * frame_interval)
        idx2 = min(idx1 + 1, max_index)

        # Find the interpolation weight
        alpha = (i / length * frame_interval) - idx1

        # Interpolate between the original frames
        cam1 = cv2.resize(cam[idx1], (w, h))
        cam2 = cv2.resize(cam[idx2], (w, h))
        grayscale_cam_resized[i] = cv2.addWeighted(cam1, 1 - alpha, cam2, alpha, 0)

    return grayscale_cam_resized[:interval, :, :]


def naive_overlap_cams(cam: np.ndarray, target_length, h:int=512, w:int=512) -> np.ndarray:
    assert cam.shape[0] == 2, f"cam.shape[0]={cam.shape[0]} != 2"

    _cam = np.zeros((target_length, h, w))
    start_idx = target_length - 40
    overlap = 40 - start_idx

    _cam[:start_idx] = cam[0][:start_idx]
    _cam[start_idx:40] = (cam[0][start_idx:40] + cam[1][:overlap]) / 2
    _cam[40:] = cam[1][80 - target_length:]

    return _cam


def interpolate_cam_on_voxel(voxel: np.ndarray, cam: np.ndarray, conversion_info: List[Dict])-> np.ndarray:
    assert len(conversion_info) == cam.shape[
        0], f"len(conversion_info)={len(conversion_info)} != cam.shape[0]={cam.shape[0]}"

    slice_indicator = np.ones(voxel.shape[0])
    final_mask = np.zeros(voxel.shape)

    # parse conversion info
    for i in range(len(conversion_info)):
        info = conversion_info[i]
        _type = info["type"]
        _total_slice = info["total_slice"]
        _current_index = info["current_index"]
        _z1 = info["z1"]
        _z2 = info["z2"]
        _x1 = info["x1"]
        _x2 = info["x2"]
        _y1 = info["y1"]
        _y2 = info["y2"]
        real_length = _z2 - _z1
        real_height = _y2 - _y1
        real_width = _x2 - _x1

        if _type == "overlap":
            i += 1
            former = cam_to_intermediate_cam(cam[i], 40, 40, real_height, real_width)
            latter = cam_to_intermediate_cam(cam[i + 1], 40, 40, real_height, real_width)
            current_cam = naive_overlap_cams(np.stack([former, latter]), _total_slice, real_height, real_width)
            current_cam = cam_to_intermediate_cam(current_cam, real_length, real_length, real_height, real_width)
        elif _type == "padding":
            slice_indicator[_z1:_z2] = 1
            current_cam = cam_to_intermediate_cam(cam[i], int(40 * real_length / _total_slice), real_length, real_height, real_width)
        elif _type == "normal":
            slice_indicator[_z1:_z2] = 1
            current_cam = cam_to_intermediate_cam(cam[i], real_length, real_length, real_height, real_width)
        else:
            continue

        slice_indicator[_z1:_z2] += 1
        # DEBUG use
        # if np.max(current_cam) < 0.5:
        #     current_cam = np.ones_like(current_cam)
        final_mask[_z1:_z2, _y1:_y2, _x1:_x2] += current_cam

    # No need of normalization
    return final_mask

def convert_to_rgb(image):
    return np.stack((image,) * 3, axis=-1)