import os

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder


def encode(s: pd.Series):
    unique_values = np.unique(s)
    mapping = {}
    for item in s:
        if item not in mapping:
            mapping[item] = len(mapping)
        if len(mapping) == len(unique_values):
            break
    s = s.apply(lambda x: mapping[x])
    return s



if __name__ == '__main__':
    datasets = [
        # "FONTS",
        # "WISDM",
        # "SENSORS",
        "GAS",
        # "COVTYPE",
        # "PAMAP2",
        # "HARTH",
        # "KDD",
        # "SUSY",
        # "RIALTO",
        # "AIRLINES",
        # "INSECTS",
        # "POKER",
        # "HEPMASS",
        # "HIGGS",
    ]

    columns = ["Dataset", "Correlation"]
    correlations = []
    for dataset in datasets:
        ds_path = os.path.join(os.getcwd(), "data", "processed", dataset, "unshuffled", dataset + ".arff")
        df = pd.DataFrame(loadarff(ds_path)[0])
        target_df = df[["class"]]
        target_df.loc[:, "class"] = encode(target_df["class"])
        target_df.loc[:, "class"] = target_df["class"].astype(int)
        target_df.loc[:, "index"] = np.arange(len(target_df))
        print(dataset)
        print(target_df.corr())