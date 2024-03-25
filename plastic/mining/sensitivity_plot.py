import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plastic.mining.create_plots import plot
from util import get_palette, get_all_approaches

from critdd import Diagram


def add_diff_column(df: pd.DataFrame, column_to_differentiate) -> pd.Series:
    s = df[column_to_differentiate].diff()
    return s


if __name__ == '__main__':
    METRIC = "classifications correct (percent)"
    DATASET_COL = "Dataset"
    APPROACH_COL = "Approach"
    ORDER_COL = "Order"
    NUM_CATEGORICAL = "\# Categories"
    MAX_BRANCH_LENGTH = r"$\nu$"

    dataframes = []
    main_result_dir = os.path.join(os.getcwd(), "results", "sensitivity_study")
    for dataset_name in os.listdir(main_result_dir):
        filepath = os.path.join(main_result_dir, dataset_name)
        this_df = pd.read_parquet(filepath)

        this_df[MAX_BRANCH_LENGTH] = this_df[APPROACH_COL].apply(lambda x: int(x.split(" (")[-1].split(")")[0]))
        this_df[NUM_CATEGORICAL] = int(dataset_name.split(" (")[-1].split(")")[0])
        this_df[DATASET_COL] = os.path.splitext(dataset_name)[0].split(" (")[0]
        this_df[APPROACH_COL] = this_df[APPROACH_COL].apply(lambda x: os.path.splitext(x)[0].split(" (")[0])

        dataframes.append(this_df)

    dataframe = pd.concat(dataframes, ignore_index=True)
    dataframe = (dataframe
                 .groupby([APPROACH_COL, DATASET_COL, NUM_CATEGORICAL, MAX_BRANCH_LENGTH])
                 .mean()
                 .reset_index()
                 )

    grid = sns.relplot(dataframe, x=MAX_BRANCH_LENGTH, y=METRIC, col=NUM_CATEGORICAL,
                       kind="line", markers="o", facet_kws={"sharey": False})
    plt.show()

