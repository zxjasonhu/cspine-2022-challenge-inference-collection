import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np


studies = glob.glob("../../data/train-numpy-seg-pngs/*")
labels = [s.replace("train", "label") for s in studies]

for s,l in zip(studies, labels):
    img = cv2.imread(s, 0)
    seg = cv2.imread(l, cv2.IMREAD_UNCHANGED)
    print(np.unique(seg))
    seg = seg.astype("float32") / 8.0
    seg *= 255
    seg = seg.astype("uint8")
    plt.subplot(1,2,1)
    plt.imshow(img, cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(seg, cmap="gray")
    plt.show()