import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from util import df_to_arff

if __name__ == '__main__':
    n_shuffles = 10
    dataset_name = "HARTH"
    path_to_raw_files = os.path.join(os.getcwd(), "data", "raw", "harth")
    path_to_processed_files = os.path.join(os.getcwd(), "data", "processed", dataset_name)

    all_dfs = []
    for i, file in enumerate(os.listdir(path_to_raw_files)):
        fp = os.path.join(path_to_raw_files, file)
        df = pd.read_csv(fp,
                         index_col=None,
                         delimiter=",",
                         dtype={"back_x": float, "back_y": float, "back_z": float,
                                "thigh_x": float, "thigh_y": float, "thigh_z": float,
                                "label": "category"})
        df.ffill(inplace=True)
        df.bfill(inplace=True)  #ffill does not work if there are nans at the start of the file, so we also do bfill afterwards.
        df.rename({"label": "class"}, axis=1, inplace=True)
        df = df[["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z", "class"]]
        all_dfs.append(df)

    unshuffled_df = pd.concat(all_dfs, ignore_index=True)
    unshuffled_df["class"] = unshuffled_df["class"].astype("category")
    unshuffled_df = pd.concat([unshuffled_df.drop("class", axis=1), unshuffled_df["class"]], axis=1)
    df_to_arff(unshuffled_df, os.path.join(path_to_processed_files, "unshuffled", dataset_name + ".arff"))

    for i in tqdm(range(10)):
        shuffled_df = unshuffled_df.sample(frac=1, replace=False, random_state=i)
        df_to_arff(shuffled_df, os.path.join(path_to_processed_files, "shuffled", dataset_name + f"-{i}" + ".arff"))
