import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd

from utils import overlay_images


PNG_DIR = "../../data/pngs/"

df = pd.read_csv("../../data/train_bounding_boxes.csv")
df["filename"] = df.StudyInstanceUID + "/" + df.slice_number.astype("str") + ".png"
df = df.sample(n=len(df))

for row_num, row in df.iterrows():
    img = cv2.imread(osp.join(PNG_DIR, row.filename), cv2.IMREAD_UNCHANGED)
    x, y, w, h = row.x, row.y, row.width, row.height
    x, y, w, h = int(x), int(y), int(w), int(h)
    mask = np.zeros_like(img)
    mask[y:y+h, x:x+w] = 255
    overlaid = overlay_images(img, mask)
    plt.imshow(overlaid, cmap="gray")
    plt.show()