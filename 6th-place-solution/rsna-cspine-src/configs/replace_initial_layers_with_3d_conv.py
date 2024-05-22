import numpy as np
import pandas as pd
import torch
import torch.nn as nn


NUM_IMAGES = 5
assert NUM_IMAGES % 2 == 1

df = pd.read_csv("../data/train_metadata.csv")

df_list = []
for study_id, study_df in df.groupby("StudyInstanceUID"):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    pad_num = NUM_IMAGES // 2
    files = study_df.filename.tolist()
    original_len = len(files)
    files = [files[0] * pad_num] + files + [files[-1] * pad_num]
    list_of_files = []
    for i in range(NUM_IMAGES):
        list_of_files.append(files[i:i+original_len])
    filename_2dc = []
    for i in range(original_len):
        filename_2dc.append(",".join(list_of_files[j][i] for j in range(NUM_IMAGES)))
    study_df["filename_2dc"] = filename_2dc
    for f in filename_2dc: print(f)
    df_list.append(study_df)
    import time ; time.sleep(100)
x = torch.from_numpy(np.ones((1, 1, 5, 256, 256))).float()
for i in range(2, 5):
    x[:, :, i] *= i

print(torch.unique(x))

layer = nn.Conv3d(1, 48, kernel_size=(5, 1, 1), stride=(5, 1, 1), padding=0, bias=False)
dummy_weight = torch.ones_like(layer.weight) / layer.kernel_size[0]
layer.weight = nn.Parameter(dummy_weight)

out = layer(x)
