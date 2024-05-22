import os
import pandas as pd

from tqdm import tqdm


os.makedirs("../../data/fractures/C1")


df = pd.read_csv("../../data/train.csv")
c1 = df[df.C1 == 1].StudyInstanceUID.tolist()

for study in tqdm(c1):
    os.system(f"cp -r ../../data/pngs/{study} ../../data/fractures/C1")