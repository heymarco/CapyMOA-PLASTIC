import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from util import df_to_arff

if __name__ == '__main__':
    n_shuffles = 10
    dataset_name = "WISDM"
    path_to_raw_file = os.path.join(os.getcwd(), "data", "raw", "WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")
    path_to_processed_files = os.path.join(os.getcwd(), "data", "processed", dataset_name)

    unshuffled_df = pd.read_csv(path_to_raw_file, delimiter=",", header=None,
                                names=["user", "activity", "timestamp", "x-accel", "y-accel", "z-accel"],
                                on_bad_lines="skip", dtype={"user": "category", "activity": "category"})
    unshuffled_df["activity"] = LabelEncoder().fit_transform(unshuffled_df["activity"])
    unshuffled_df["activity"] = unshuffled_df["activity"].astype("category")
    unshuffled_df.drop(["timestamp"], axis=1, inplace=True)
    unshuffled_df.dropna(axis=0)
    unshuffled_df.rename({"activity": "class"}, axis=1, inplace=True)
    unshuffled_df["z-accel"] = unshuffled_df["z-accel"].apply(lambda x: float(x[:-1]))
    unshuffled_df.astype({"z-accel": float})
    unshuffled_df = pd.concat([unshuffled_df.drop("class", axis=1), unshuffled_df["class"]], axis=1)

    df_to_arff(unshuffled_df, os.path.join(path_to_processed_files, "unshuffled", dataset_name + ".arff"))

    for i in tqdm(range(10)):
        shuffled_df = unshuffled_df.sample(frac=1, replace=False, random_state=i)
        df_to_arff(shuffled_df, os.path.join(path_to_processed_files, "shuffled", dataset_name + f"-{i}" + ".arff"))