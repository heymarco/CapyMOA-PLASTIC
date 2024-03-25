import os

import pandas as pd
from tqdm import tqdm

from util import df_to_arff

if __name__ == '__main__':
    n_shuffles = 10
    dataset_name = "POKER"
    path_to_raw_file = os.path.join(os.getcwd(), "data", "raw", "poker+hand", "poker-hand-testing.data")
    path_to_processed_files = os.path.join(os.getcwd(), "data", "processed", dataset_name)

    unshuffled_df = pd.read_csv(path_to_raw_file, header=None,
                                on_bad_lines="skip", dtype="category")
    unshuffled_df.dropna(axis=0)
    unshuffled_df.rename({10: "class"}, inplace=True, axis=1)
    unshuffled_df = pd.concat([unshuffled_df.drop("class", axis=1), unshuffled_df["class"]], axis=1)

    df_to_arff(unshuffled_df, os.path.join(path_to_processed_files, "unshuffled", dataset_name + ".arff"))

    for i in tqdm(range(10)):
        shuffled_df = unshuffled_df.sample(frac=1, replace=False, random_state=i)
        df_to_arff(shuffled_df, os.path.join(path_to_processed_files, "shuffled", dataset_name + f"-{i}" + ".arff"))