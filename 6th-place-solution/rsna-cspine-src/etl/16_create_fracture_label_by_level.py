import pandas as pd

from tqdm import tqdm


df = pd.read_csv("../../data/train_cas_kfold_all.csv")
train_df = pd.read_csv("../../data/train.csv")
# Create dictionary mapping study ID to fractured vertebra for easier access
train_dict = {}
for rownum, row in train_df.iterrows():
    train_dict[row.StudyInstanceUID] = row.index[row == 1].tolist()


df_list = []
for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    for level in [f"C{i+1}" for i in range(7)]:
        study_df[f"{level}_frac"] = 0
        if level in train_dict[study_id]:
            study_df.loc[(study_df.fracture_cas == 1) & (study_df[f"{level}_present"] == 1), f"{level}_frac"] = 1
    df_list.append(study_df)

df = pd.concat(df_list)
for level in [f"C{i+1}_frac" for i in range(7)]:
    print(df[level].value_counts())


df.to_csv("../../data/train_cas_kfold_all_by_level.csv", index=False)