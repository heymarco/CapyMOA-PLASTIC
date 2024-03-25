import os

import pandas as pd
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo

from util import df_to_arff

if __name__ == '__main__':
    n_shuffles = 10
    dataset_name = "COVTYPE"
    path_to_processed_files = os.path.join(os.getcwd(), "data", "processed", dataset_name)

    # fetch dataset
    covertype = fetch_ucirepo(id=31)

    # data (as pandas dataframes)
    X = covertype.data.features
    y = covertype.data.targets

    unshuffled_df = pd.concat([X, y], axis=1).reset_index()
    unshuffled_df.dropna(axis=0)
    unshuffled_df.rename({"Cover_Type": "class"}, axis=1, inplace=True)

    for col in unshuffled_df.columns:
        if "Soil_Type" in str(col) or "class" == str(col) or "Wilderness_Area" == str(col):
            unshuffled_df[col] = unshuffled_df[col].astype("category")

    unshuffled_df = pd.concat([unshuffled_df.drop("class", axis=1), unshuffled_df["class"]], axis=1)

    df_to_arff(unshuffled_df, os.path.join(path_to_processed_files, "unshuffled", dataset_name + ".arff"))

    for i in tqdm(range(10)):
        shuffled_df = unshuffled_df.sample(frac=1, replace=False, random_state=i)
        df_to_arff(shuffled_df, os.path.join(path_to_processed_files, "shuffled", dataset_name + f"-{i}" + ".arff"))