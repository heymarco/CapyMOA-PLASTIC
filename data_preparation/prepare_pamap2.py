import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from util import df_to_arff

if __name__ == '__main__':
    n_shuffles = 10
    dataset_name = "PAMAP2"
    path_to_raw_files = os.path.join(os.getcwd(), "data", "raw", "PAMAP2_Dataset", "Protocol")
    path_to_processed_files = os.path.join(os.getcwd(), "data", "processed", dataset_name)

    all_dfs = []
    for file in os.listdir(path_to_raw_files):
        fp = os.path.join(path_to_raw_files, file)
        df = pd.read_csv(fp, delimiter=" ", header=None, dtype={1: "category"})
        df.ffill(inplace=True)
        df.bfill(inplace=True)  #ffill does not work if there are nans at the start of the file, so we also do bfill afterwards.
        df.drop([0], axis=1, inplace=True)
        df.rename({1: "class"}, axis=1, inplace=True)
        all_dfs.append(df)

    unshuffled_df = pd.concat(all_dfs, ignore_index=True)
    unshuffled_df["class"] = unshuffled_df["class"].astype("category")
    unshuffled_df = pd.concat([unshuffled_df.drop("class", axis=1), unshuffled_df["class"]], axis=1)
    df_to_arff(unshuffled_df, os.path.join(path_to_processed_files, "unshuffled", dataset_name + ".arff"))

    for i in tqdm(range(10)):
        shuffled_df = unshuffled_df.sample(frac=1, replace=False, random_state=i)
        df_to_arff(shuffled_df, os.path.join(path_to_processed_files, "shuffled", dataset_name + f"-{i}" + ".arff"))
