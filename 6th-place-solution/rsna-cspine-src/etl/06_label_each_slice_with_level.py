import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd

from collections import defaultdict
from tqdm import tqdm


studies = glob.glob("../../data/pngs-segmentations/*")
df_dict = defaultdict(list)
map_dict = {31: 1, 63: 2, 95: 3, 127: 4, 159: 5, 191: 6, 223: 7}

for s in tqdm(studies, total=len(studies)):
    images = glob.glob(osp.join(s, "*png"))
    for i in images:
        df_dict["filename"].append(i)
        img = cv2.imread(i, 0)
        levels = np.zeros((7,))
        img_levels = np.unique(img)
        for each_level in img_levels:
            if each_level > 0 and each_level < 255:
                levels[map_dict[each_level] - 1] = 1
        df_dict["levels"].append(levels)


levels_array = np.asarray(df_dict["levels"]).astype("bool")
levels_df = pd.DataFrame(levels_array)
levels_df.columns = [f"C{i+1}" for i in range(7)]
levels_df["filename"] = df_dict["filename"]
levels_df["filename"] = levels_df.filename.apply(lambda x: x.replace("../../data/", ""))
levels_df["StudyInstanceUID"] = levels_df.filename.apply(lambda x: x.split("/")[-2])
levels_df["slice_number"] = levels_df.filename.apply(lambda x: x.split("/")[-1].replace(".png", "")).astype("int")
levels_df = levels_df.sort_values(["StudyInstanceUID", "slice_number"], ascending=[True, False])
levels_df.to_csv("../../data/train_cspine_levels.csv", index=False)