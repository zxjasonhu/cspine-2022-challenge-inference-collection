import pandas as pd


NUM_IMAGES = 5
assert NUM_IMAGES % 2 == 1

df = pd.read_csv("../../data/train_metadata.csv")

df_list = []
for study_id, study_df in df.groupby("StudyInstanceUID"):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    pad_num = NUM_IMAGES // 2
    files = study_df.filename.tolist()
    files = [f.replace("dcm", "png") for f in files]
    original_len = len(files)
    files = [files[0]] * pad_num + files + [files[-1]] * pad_num
    list_of_files = []
    for i in range(NUM_IMAGES):
        list_of_files.append(files[i:i+original_len])
    filename_2dc = []
    for i in range(original_len):
        filename_2dc.append(",".join(list_of_files[j][i] for j in range(NUM_IMAGES)))
    study_df["filename_2dc"] = filename_2dc
    df_list.append(study_df)

pd.concat(df_list).to_csv("../../data/train_metadata_2dc.csv", index=False)