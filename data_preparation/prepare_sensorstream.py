import os

import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from util import df_to_arff

if __name__ == '__main__':
    n_shuffles = 10
    dataset_name = "SENSORS"
    path_to_raw_file = os.path.join(os.getcwd(), "data", "raw", "SENSORS", "sensorstream.arff")
    path_to_processed_files = os.path.join(os.getcwd(), "data", "processed", dataset_name)

    unshuffled_df = pd.DataFrame(loadarff(path_to_raw_file)[0])

    unshuffled_df.dropna(axis=0)
    unshuffled_df = pd.concat([unshuffled_df.drop("class", axis=1), unshuffled_df["class"]], axis=1)
    unshuffled_df["class"] = LabelEncoder().fit_transform(unshuffled_df["class"])
    unshuffled_df["class"] = unshuffled_df["class"].astype(str).astype("category")

    df_to_arff(unshuffled_df, os.path.join(path_to_processed_files, "unshuffled", dataset_name + ".arff"))

    for i in tqdm(range(10)):
        shuffled_df = unshuffled_df.sample(frac=1, replace=False, random_state=i)
        df_to_arff(shuffled_df, os.path.join(path_to_processed_files, "shuffled", dataset_name + f"-{i}" + ".arff"))