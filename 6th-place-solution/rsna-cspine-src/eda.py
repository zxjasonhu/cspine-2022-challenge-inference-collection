import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import torch


def plot_study(stack, num_row=4, num_col=4, sagittal=True):
    stack_shape = stack.shape[2] if sagittal else stack.shape[0]
    index = 1
    for i in np.linspace(0, stack_shape - 1, num_row * num_col).astype("int"):
        plt.subplot(num_row, num_col, index)
        plt.imshow(stack[..., i] if sagittal else stack[i], cmap="gray")
        index += 1
    plt.show()


studies = np.sort(glob.glob("../data/train-numpy-vertebra-chunks-with-20-buffer/*"))
study = np.load(studies[1])
print(study.shape)
plot_study(study, sagittal=False)



