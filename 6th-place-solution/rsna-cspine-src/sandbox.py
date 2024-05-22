import glob
import matplotlib.pyplot as plt
import numpy as np


def plot_volume(img, num_row, num_col, sagittal=True):
    index = 1
    img_shape = img.shape[2] if sagittal else img.shape[0]
    for i in np.linspace(0, img_shape - 1, num_row * num_col).astype("int"):
        plt.subplot(num_row, num_col, index)
        plt.imshow(img[:, :, i] if sagittal else img[i], cmap="gray")
        index += 1
    plt.show()


x = glob.glob("../data/train-individual-vertebrae-masked/*.npy")
for i in x:
    plot_volume(np.load(i), 6, 4, sagittal=False)