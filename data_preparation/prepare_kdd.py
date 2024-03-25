import os

import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from util import df_to_arff

if __name__ == '__main__':
    n_shuffles = 10
    dataset_name = "KDD"
    path_to_raw_file = os.path.join(os.getcwd(), "data", "raw", "KDD", "kddcup.data")
    path_to_processed_files = os.path.join(os.getcwd(), "data", "processed", dataset_name)

    names = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "class"]

    unshuffled_df = pd.read_csv(path_to_raw_file,
                                header=None,
                                names=names,
                                sep=",",
                                dtype={"protocol_type": "category", "service": "category", "flag": "category",
                                       "land": "category", "logged_in": "category",
                                       "is_host_login": "category", "is_guest_login": "category",
                                       "class": "category"})
    for col in unshuffled_df.columns:
        if unshuffled_df[col].dtype == "category":
            unshuffled_df[col] = LabelEncoder().fit_transform(unshuffled_df[col])
            unshuffled_df[col] = unshuffled_df[col].astype(str).astype("category")

    unshuffled_df.dropna(axis=0)
    unshuffled_df = pd.concat([unshuffled_df.drop("class", axis=1), unshuffled_df["class"]], axis=1)

    df_to_arff(unshuffled_df, os.path.join(path_to_processed_files, "unshuffled", dataset_name + ".arff"))

    for i in tqdm(range(10)):
        shuffled_df = unshuffled_df.sample(frac=1, replace=False, random_state=i)
        df_to_arff(shuffled_df, os.path.join(path_to_processed_files, "shuffled", dataset_name + f"-{i}" + ".arff"))
