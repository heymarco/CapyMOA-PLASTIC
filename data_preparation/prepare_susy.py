import os

import pandas as pd
from tqdm import tqdm

from util import df_to_arff

if __name__ == '__main__':
    n_shuffles = 10
    dataset_name = "SUSY"
    path_to_raw_file = os.path.join(os.getcwd(), "data", "raw", "SUSY", "SUSY.csv")
    path_to_processed_files = os.path.join(os.getcwd(), "data", "processed", dataset_name)

    columns = ["class", "lepton__1_pT", "lepton__1_eta", "lepton__1_phi", "lepton__2_pT", "lepton__2_eta",
               "lepton__2_phi", "missing_energy_magnitude", "missing_energy_phi", "MET_rel", "axial_MET", "M_R",
               "M_TR_2", "R", "MT2", "S_R", "M_Delta_R", "dPhi_r_b", "cos(theta_r1)"]
    dtypes = {"class": "category"}

    unshuffled_df = pd.read_csv(path_to_raw_file, header=None,
                                names=columns,
                                on_bad_lines="skip", dtype=dtypes)
    unshuffled_df.dropna(axis=0)
    unshuffled_df = pd.concat([unshuffled_df.drop("class", axis=1), unshuffled_df["class"]], axis=1)

    df_to_arff(unshuffled_df, os.path.join(path_to_processed_files, "unshuffled", dataset_name + ".arff"))

    # for i in tqdm(range(10)):
    #     shuffled_df = unshuffled_df.sample(frac=1, replace=False, random_state=i)
    #     df_to_arff(shuffled_df, os.path.join(path_to_processed_files, "shuffled", dataset_name + f"-{i}" + ".arff"))