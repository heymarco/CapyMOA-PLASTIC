import os

import pandas as pd

if __name__ == '__main__':
    old_dir = os.path.join(os.getcwd(), "results", "processed_submission")
    target_dir = os.path.join(os.getcwd(), "results", "processed_submission2")

    datasets = [
        "WISDM",
        "SENSORS",
        "GAS",
        "COVTYPE",
        "PAMAP2",
        "HARTH",
        "KDD",
        "SUSY",
        "RIALTO",
        # "FONTS",
        "AIRLINES",
        "INSECTS",
        "POKER",
        "HEPMASS",
        "HIGGS",
    ]

    keep_approaches = [
        "NoChange",
        # "PLASTIC"
        # "PLASTIC-c",
        "EFHAT", "HT", "EFDT"
    ]

    for dataset in os.listdir(old_dir):
        ds_name, ext = os.path.splitext(dataset)
        ds_name = ds_name.split("_")[0]
        if ds_name in datasets:
            path = os.path.join(old_dir, dataset, "unshuffled", ds_name + ".parquet")
            target_path = os.path.join(target_dir, ds_name + "_competitors", "unshuffled")
            df = pd.read_parquet(path)
            df = df.loc[df["Approach"].isin(keep_approaches)]
            if len(df) > 0:
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                df.to_parquet(os.path.join(target_path, ds_name + ".parquet"))
