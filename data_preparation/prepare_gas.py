import io
import os

import pandas as pd
from tqdm import tqdm

from util import df_to_arff

if __name__ == '__main__':
    n_shuffles = 10
    dataset_name = "GAS"
    path_to_raw_files = os.path.join(os.getcwd(), "data", "raw", "gas+sensor+array+under+dynamic+gas+mixtures")
    path_to_processed_files = os.path.join(os.getcwd(), "data", "processed", dataset_name)

    columns = ['time', 'gas1', 'gas2'] + [f'Reading_{i + 1}' for i in range(16)]
    dataframes = []

    path_to_processed_files_unshuffled = os.path.join(path_to_processed_files, "unshuffled")
    if not os.path.exists(path_to_processed_files_unshuffled):
        os.makedirs(path_to_processed_files_unshuffled)

    create = True
    content = ""
    output_path = os.path.join(path_to_processed_files_unshuffled, dataset_name + ".arff")
    for i, filename in enumerate(os.listdir(path_to_raw_files)):
        with open(os.path.join(path_to_raw_files, filename), 'r') as file:
            for line_index, line in enumerate(file):
                sep = ","
                line = sep.join(line.split())
                if line_index == 0:
                    continue
                content = f"{content}{line}\n"
                if line_index % 10000 == 10000 - 1:
                    newfile = io.StringIO(content)
                    df = pd.read_csv(newfile, delimiter=sep, names=columns, dtype={})
                    df["class"] = "0"
                    # 0 if ethylene has the highest concentration, 1 or 2 if the other if CO or methane have the highest concentrations
                    df.loc[df["gas1"] > df["gas2"], "class"] = str(i + 1)
                    df["class"] = df["class"].astype("category")
                    df = df.drop(["gas1", "gas2", "time"], axis=1)
                    if create:
                        attributes = [(str(j), 'NUMERIC') if df[j].dtypes not in ['str', "category"]
                                      else (str(j), ["0", "1", "2"]) for j in df]
                        df_to_arff(df, output_path, attributes)
                        create = False
                    else:
                        df.to_csv(output_path, mode="w" if create else "a",
                                  header=not os.path.exists(output_path), index=False)
                    content = ""
            # get the last lines that were not yet collected
            newfile = io.StringIO(content)
            df = pd.read_csv(newfile, delimiter=sep, names=columns, dtype={})
            df["class"] = "0"
            # 0 if ethylene has the highest concentration, 1 or 2 if the other if CO or methane have the highest concentrations
            df.loc[df["gas1"] > df["gas2"], "class"] = str(i + 1)
            df["class"] = df["class"].astype("category")
            df = df.drop(["gas1", "gas2", "time"], axis=1)
            df.to_csv(output_path, mode="w" if create else "a",
                      header=not os.path.exists(output_path), index=False)
            create = False

    #
    #     df = pd.read_csv(newfile, delimiter=sep, names=columns, dtype={})
    #     df = df.iloc[::10]
    #
    #     dataframes.append(df)
    #
    # unshuffled_df = pd.concat(dataframes, ignore_index=True)
    # unshuffled_df.dropna(axis=0)
    # df_to_arff(unshuffled_df, os.path.join(path_to_processed_files, "unshuffled", dataset_name + ".arff"))

    # for i in tqdm(range(10)):
    #     shuffled_df = unshuffled_df.sample(frac=1, replace=False, random_state=i)
    #     df_to_arff(shuffled_df, os.path.join(path_to_processed_files, "shuffled", dataset_name + f"-{i}" + ".arff"))
